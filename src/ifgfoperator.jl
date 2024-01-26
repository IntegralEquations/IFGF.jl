"""
    struct MulBuffer{T}

Low-level structure used to hold the interpolation coefficients and values of
the current and next level cone interpolants.

The type paramter `T` is the type of the interpolant.
"""
struct MulBuffer{T}
    current_coefs::Vector{T}
    next_coefs::Vector{T}
    plan::FFTW.FFTWPlan
end

"""
    struct IFGFOp{T} <: AbstractMatrix{T}

High-level structure representing a linear operator with elements of type `T`
with entry `i,j` given by `K(X[i],Y[j])`, where `K` is the underlying kernel
function, and `X` and `Y` are the elements of the `target_tree` and
`source_tree`, respectively.

# Fields:
- `kernel` : underlying kernel function
- `target_tree` : tree structure used for clustering the target points
- `source_tree` : tree structure used for clustering the source points
- `p::NTuple`: number of chebyshev poins per dimension used in the interpolation
  of far interactions
- `ds_func` : function used to compute and appropriate mesh size in parameter
  space. See e.g. [`cone_domain_size_func`](@ref)
- `adm_func` : function used to determine if the block with targets on box `A`
  and sources on box `B` admit a low-rank approximation
- `_source_tree_depth_partition` : vector with `i`-th entry containing all nodes
  of the of depth `i-1`, where the root is given a depth of `0`.
"""
struct IFGFOp{T} <: AbstractMatrix{T}
    kernel
    target_tree::TargetTree
    source_tree::SourceTree
    p::NTuple
    ds_func::Function
    adm_func::Function
    _source_tree_depth_partition
    mulbuffers::Ref
end

# getters
rowrange(ifgf::IFGFOp)         = index_range(ifgf.target_tree)
colrange(ifgf::IFGFOp)         = index_range(ifgf.source_tree)
kernel(op::IFGFOp)             = op.kernel
target_tree(op::IFGFOp)        = op.target_tree
source_tree(op::IFGFOp)        = op.source_tree
polynomial_order(op::IFGFOp)   = op.p
meshsize_func(op::IFGFOp)      = op.ds_func
admissibility_func(op::IFGFOp) = op.adm_func

function Base.getproperty(A::IFGFOp,s::Symbol)
    if s == :plan
        return A.mulbuffers[].plan
    elseif s == :buffers
        return (A.mulbuffers[].current_coefs, A.mulbuffers[].next_coefs)
    else
        getfield(A,s)
    end
end

function Base.size(op::IFGFOp)
    sizeX = op |> target_tree |> root_elements |> length
    sizeY = op |> source_tree |> root_elements |> length
    return (sizeX, sizeY)
end

# disable getindex
function Base.getindex(::IFGFOp, args...)
    error("`getindex(::IFGFOp,args...)` has been intentionally disabled")
end

"""
    assemble_ifgf(K,X,Y;nmax=100,adm=modified_admissibility_condition,order=nothing,tol=1e-4,Δs=1.0,threads=true)

Construct an approximation of the matrix `[K(x,y) for x ∈ X, y ∈ Y]`. If the
kernel function `K` is oscillatory, you should implement the method
`wavenumber(::typeof(K)) --> Number` to return its wavenumber.

The error in the approximation is controlled by two parameters: `Δs` and `order`.
These parameters dictate the mesh size and interpolation order of the parametric
space used to build the cone interpolants. While the `order` is held constant
across all source cells, the `Δs` parameter controls only the size of the *low-frequency*
cells; for cells of large acoustic size, the mesh size is adapted so as to keep
the interpolation error constant across different levels of the source tree (see
[`cone_domain_size_func`](@ref) for implementation details).

If the keyword `order` is ommited, a tolerance `tol` must be passed. It will
then be used to estimate an appropriate interpolation `order`.

`nmax` can be used to control the maximum number of points contained in the
target/source trees.

The `adm` function gives a criterion to determine if the interaction between a
target cell and a source cell should be computed using an interpolation scheme.
"""
function assemble_ifgf(kernel, Xpoints, Ypoints;
    nmax    = 100,
    adm     = modified_admissibility_condition,
    tol     = 1e-4,
    order   = nothing,
    Δs      = 1.0,
    threads = true,
    splitter= DyadicSplitter(;nmax)
    )

    k        = wavenumber(kernel)
    ds_func = cone_domain_size_func(k, Δs)
    # infer the type of the matrix, then compute the type of density to be
    # interpolated
    @assert tol >= 0 "`tol` must be a positive real number"
    U, V = eltype(Xpoints), eltype(Ypoints)
    Tk = return_type(kernel, U, V) # kernel type
    isconcretetype(Tk) || error("unable to infer a concrete type for the kernel")
    # create source and target trees
    @timeit_debug "source tree initialization" begin
        source_tree = initialize_source_tree(; points=Ypoints, splitter)
    end
    partition = partition_by_depth(source_tree)
    @timeit_debug "target tree initialization" begin
        target_tree = initialize_target_tree(; Xpoints, splitter)
    end
    # if `order` is not passed explicitly, estimate it
    if order === nothing
        order = estimate_interpolation_order(kernel, partition, ds_func, tol, (2, 2, 2))
    end
    p = order .+ 1
    @timeit_debug "interaction list computation" begin
        _compute_interaction_list!(source_tree, target_tree, adm)
    end
    @timeit_debug "cone list computation" begin
        _compute_cone_list!(partition, Xpoints, Val(p), ds_func, threads)
    end
    # call main constructor
    ifgf = IFGFOp{Tk}(kernel, target_tree, source_tree, p, ds_func, adm, partition, Ref{Any}(nothing))
    return ifgf
end

"""
    compute_interaction_list!(ifgf::IFGFOperator,adm)

For each node in `source_tree(ifgf)`, compute its `nearlist` and `farlist`. The
`nearlist` contains all nodes in `target_tree(ifgf)` for which the interaction must be
computed using the direct summation.  The `farlist` includes all nodes in
`target_tree` for which an accurate interpolant in available in the source node.

The `adm` function is called through `adm(target,source)` to determine if
`target` is in the `farlist` of source. When that is not the case we recurse on
their children unless both are a leaves, in which case `target` is added to the
`nearlist` of source.
"""
function compute_interaction_list!(ifgf::IFGFOp, threads=false)
    # TODO: threaded implementation of this pass
    _compute_interaction_list!(source_tree(ifgf), target_tree(ifgf), admissibility_func(ifgf))
    return ifgf
end

function _compute_interaction_list!(source::SourceTree, target::TargetTree, adm)
    # if either box is empty return immediately
    if length(source) == 0 || length(target) == 0
        return nothing
    end
    # handle the various possibilities
    if adm(target, source)
        _addtofarlist!(source, target)
    elseif isleaf(target) && isleaf(source)
        _addtonearlist!(source, target)
    elseif isleaf(target)
        # recurse on children of source
        for source_child in children(source)
            _compute_interaction_list!(source_child, target, adm)
        end
    elseif isleaf(source)
        # recurse on children of target
        for target_child in children(target)
            _compute_interaction_list!(source, target_child, adm)
        end
    else
        # TODO: when both have children, how to recurse down? Two strategies
        # below:
        # strategy 1: recurse on children of largest box
        if radius(source) > radius(target)
            for source_child in children(source)
                _compute_interaction_list!(source_child, target, adm)
            end
        else
            for target_child in children(target)
                _compute_interaction_list!(source, target_child, adm)
            end
        end
        # strategy 2: recurse on children of both target and source
        # for source_child in children(source)
        #     for target_child in children(target)
        #         _compute_interaction_list!(source_child, target_child, adm)
        #     end
        # end
    end

    return nothing
end

"""
    compute_cone_list!(ifgf::IFGFOperator)

For each node in `source_tree(ifgf)`, compute the cones which are necessary to cover both
the target points on its `farlist` as well as the interpolation points on its
parent's cones. The `polynomial_order(ifgf)` and `meshsize_func(ifgf)` are used
to construct the appropriate interpolation domains and points.
"""
function compute_cone_list!(ifgf::IFGFOp, threads)
    p         = polynomial_order(ifgf)
    func      = meshsize_func(ifgf)
    X         = target_tree(ifgf) |> root_elements
    partition = ifgf._source_tree_depth_partition
    _compute_cone_list!(partition, X, Val(p), func, threads)
    return ifgf
end

function _compute_cone_list!(partition, X, p, func, threads)
    for depth in partition
        if threads
            Threads.@threads for node in depth
                _initialize_cone_interpolants!(node, X, p, func)
            end
        else
            for node in depth
                _initialize_cone_interpolants!(node, X, p, func)
            end
        end
    end
    return partition
end

function _initialize_cone_interpolants!(source::SourceTree{N}, X, p::Val{P}, ds_func) where {N,P}
    length(source) == 0 && (return source)
    ds = ds_func(source)
    # find minimal axis-aligned bounding box for interpolation points
    domain = compute_interpolation_domain(source, p)
    all(low_corner(domain) .< high_corner(domain)) || (return source)
    _init_msh!(source, domain, ds)
    # if not a root, create all cones needed to cover interpolation points
    # of parents
    refnodes = chebnodes(p)
    if !isroot(source)
        source_parent = parent(source)
        for I in active_cone_idxs(source_parent)
            rec = cone_domain(source_parent, I)
            lc, hc = low_corner(rec), high_corner(rec)
            c0 = (lc + hc) / 2
            c1 = (hc - lc) / 2
            cheb = map(x -> c0 + c1 .* x, refnodes)
            # cheb = cheb2nodes(p,rec)
            for si in cheb # interpolation node in interpolation space
                # interpolation node in physical space
                xi = interp2cart(si, source_parent)
                # mesh index for interpolation point
                I, _ = cone_index(xi, source)
                # initialize cone if needed
                _addnewcone!(source, I, p)
            end
        end
    end
    # initialize all cones needed to cover far field.
    for far_target in farlist(source)
        for x in elements(far_target) # target points
            I, _ = cone_index(x, source)
            _addnewcone!(source, I, p)
        end
    end
    sort!(conedatadict(source))
    return source
end

"""
    mul!(y,A::IFGFOp,x,a,b;global_index=true,threads=true)

Like the `mul!` function from `LinearAlgebra`. Setting `global_index=false`
uses the internal indexing system of the `IFGFOp`, and therefore does not
permute the vectors `x` and `y` to perform the multiplication.
"""
function LinearAlgebra.mul!(y::AbstractVector, A::IFGFOp, x::AbstractVector, a::Number, b::Number; global_index=true, threads=true)
    partition = A._source_tree_depth_partition
    # since the IFGF represents A = Pr*K*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end.
    TA = eltype(A)
    # preallocate buffers for holding all interpolation values/coefs and decide
    # which memory is "owned" by each cone
    Tx = eltype(x)
    Ty = Base.promote_op(*,TA,Tx) # interpolant type (same as output type)
    if !isa(A.mulbuffers[],MulBuffer{Ty})
        bufs = _allocate_buffers!(partition,prod(A.p),Ty)
        # create an fftw plan for the chebtransform. Since `Ti` may be an `SVector`, we plan on
        # its element type and perform the FFT on each component
        plan = FFTW.plan_r2r!(zeros(eltype(Ty), A.p), FFTW.REDFT00; flags=FFTW.PATIENT | FFTW.UNALIGNED)
        A.mulbuffers[] = MulBuffer(bufs[1], bufs[2], plan)
    end
    rtree = target_tree(A)
    ctree = source_tree(A)
    # permute input
    if global_index
        x = x[ctree.loc2glob]
        y = permute!(y, rtree.loc2glob)
        rmul!(x, a) # multiply in place since this is a new copy, so does not mutate exterior x
    else
        x = a * x # new copy of x
    end
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    # extract the kernel and place a function barrier for heavy computation
    K = kernel(A)

    if threads
        _gemv_threaded!(y, K, rtree, partition, x, TA, Val(A.p),A.plan,A.buffers)
    else
        _gemv!(y, K, rtree, partition, x, TA, Val(A.p),A.plan,A.buffers)
    end
    # permute output
    global_index && invpermute!(y, loc2glob(rtree))
    return y
end



function LinearAlgebra.mul!(y::AbstractMatrix, A::IFGFOp, x::AbstractMatrix, a::Number, b::Number; global_index=true, threads=true)
    ncol = size(x,2)
    for i in 1:ncol
        mul!(view(y,:,i),A,view(x,:,i),a,b;global_index,threads)
    end
    return y
end



# memory need for forward map
function _allocate_buffers!(partition,p,Tv)
    nmax      = 0
    for nodes in partition
        istart = 0
        iend   = 0
        for node in nodes
            for I in active_cone_idxs(node)
                istart = iend   + 1
                iend   = istart + p - 1
                node.data.conedatadict[I] = istart:iend
            end
        end
        nmax = max(nmax,iend)
    end
    buff1 = Vector{Tv}(undef,nmax)
    buff2 = Vector{Tv}(undef,nmax)
    return (buff1,buff2)
end

function _gemv!(C, K, target_tree, partition, B, T, p::Val{P}, plan, buffers) where {P}
    buffer,cbuffer = buffers
    for depth in Iterators.reverse(partition)
        fill!(buffer,zero(eltype(buffer)))
        for node in depth
            length(node) == 0 && continue
            if isleaf(node)
                @timeit_debug "P2P" _particles_to_particles!(C, K, target_tree, node, B, node)
                @timeit_debug "P2I" _particles_to_interpolants!(node, K, B, p, plan, buffer)
            else
                @timeit_debug "I2I" _interpolants_to_interpolants(node, K, B, p, plan, buffer, cbuffer)
            end
            @timeit_debug "V2C" _chebtransform!(node, p, plan, buffer)
            @timeit_debug "I2P" _interpolants_to_particles!(C, K, target_tree, node, p, buffer)
        end
        cbuffer, buffer = buffer, cbuffer
    end
end

function _gemv_threaded!(C, K, target_tree, partition, B, T, p::Val{P}, plan, buffers) where {P}
    buffer,cbuffer = buffers
    fill!(buffer,zero(eltype(buffer)))
    fill!(buffer,zero(eltype(cbuffer)))
    nt = Threads.nthreads()
    Cthreads = [zero(C) for _ in 1:nt]
    for depth in Iterators.reverse(partition)
        fill!(buffer,zero(eltype(buffer)))
        Threads.@threads for node in depth
            id = Threads.threadid()
            length(node) == 0 && continue
            if isleaf(node)
                @timeit_debug "P2P" _particles_to_particles!(Cthreads[id], K, target_tree, node, B, node)
                @timeit_debug "P2M" _particles_to_interpolants!(node, K, B, p, plan, buffer)
            else
                @timeit_debug "I2I" _interpolants_to_interpolants(node, K, B, p, plan, buffer, cbuffer)
            end
        end
        for node in depth
            _chebtransform!(node, p, plan, buffer)
            @timeit_debug "M2P" _interpolants_to_particles!(C, K, target_tree, node, p, buffer)
        end
        cbuffer, buffer = buffer, cbuffer
    end
    # reduction stage
    for i in 1:nt
        axpy!(1, Cthreads[i], C)
    end
end

function _interpolants_to_interpolants(node::SourceTree{N,Td}, K, B, p::Val{P}, plan, buff, cbuff) where {N,Td,P}
    for I in active_cone_idxs(node)
        rec = cone_domain(node, I)
        s = chebnodes(p, rec) # cheb points in parameter space
        x = map(si -> interp2cart(si, node), s) # cheb points in physical space
        vals = reshape(view(buff,conedata(node,I)),P)
        # check if left index Il exists. If so, the data shared between Il and I
        # will be handled by Il
        idxs = if _share_interp_data()
            ntuple(N) do dim
                Il = decrement_index(I,dim)
                haskey(conedatadict(node),Il) ? (1:P[dim]-1) : (1:P[dim])
                # 1:P[dim]
            end |> CartesianIndices
        else
            eachindex(x)
        end
        for chd in children(node)
            # TODO: if number of nodes is too small, just evalute directly
            length(chd) == 0 && continue
            for i in idxs
                idxcone, si = cone_index(x[i], chd)
                # transfer block's interpolant to parent's interpolants
                rec = cone_domain(chd, idxcone)
                coefs = view(cbuff,conedata(chd,idxcone))
                vals[i] += transfer_factor(K, x[i], chd) * chebeval(coefs, si, rec, p)
            end
        end
        # copy the data to the cone on the right if it exists
        if _share_interp_data()
            _transfer_right(I,vals,buff,conedatadict(node),p)
        end
    end
    return node
end

# transfer the data from a left cell to its right neighbors
function _transfer_right(I,vals_left,buff,dict,::Val{P}) where {P}
    N = length(P)
    for dim in 1:N
        Ir = increment_index(I,dim)
        if haskey(dict,Ir)
            vals_right = reshape(view(buff,dict[Ir]),P)
            # HACK: because the line below allocates (don't know why), we have
            # to build the indices manually to copy the data.
            # copy!(selectdim(vals_right,dim,P[dim]),selectdim(vals,dim,1))
            idxs_left   = ntuple(i->i==dim ? (P[i]:P[i]) : 1:P[i],N) |> CartesianIndices
            idxs_right  = ntuple(i->i==dim ? (1:1) : 1:P[i],N) |> CartesianIndices
            for i in 1:length(idxs_left)
                vals_right[idxs_left[i]] = vals_left[idxs_right[i]]
            end
        end
    end
end

function _chebtransform!(node::SourceTree{N,Td}, ::Val{P}, plan, buff) where {N,Td,P}
    for I in active_cone_idxs(node)
        vals = view(buff,conedata(node,I))
        # transform vals to coefs in place. Coefs will be used later when the
        # current node transfers its expansion to its parent
        coefs = if _use_fftw()
            chebtransform_fftw!(reshape(vals,P...), plan)
        else
            chebtransform_native!(reshape(vals,P...))
        end
    end
    return node
end

function _particles_to_interpolants!(node::SourceTree{N,Td}, K, B, p::Val{P}, plan, buff) where {N,Td,P}
    Ypts = node |> root_elements
    for I in active_cone_idxs(node)
        rec = cone_domain(node, I)
        s = chebnodes(p, rec) # cheb points in parameter space
        x = map(si -> interp2cart(si, node), s)
        irange = 1:length(x)
        jrange = index_range(node)
        vals   = view(buff,conedata(node,I))
        near_interaction!(vals, K, x, Ypts, B, irange, jrange)
        for i in eachindex(x)
            xi = x[i] # cartesian coordinate of node
            vals[i] = inv_centered_factor(K, xi, node) * vals[i]
        end
    end
    return node
end

function _interpolants_to_particles!(C, K, target_tree, node::SourceTree{N,T}, p::Val{P}, buff) where {N,T,P}
    Xpts = target_tree |> root_elements
    for far_target in farlist(node)
        for i in index_range(far_target)
            x = Xpts[i]
            I, s = cone_index(coords(x), node)
            rec  = cone_domain(node, I)
            coefs    = view(buff,conedata(node,I))
            C[i] += centered_factor(K, Xpts[i], node) * chebeval(coefs, s, rec, p)
        end
    end
    return C
end

function _particles_to_particles!(C, K, target_tree, source_tree, B, node)
    Xpts = target_tree |> root_elements
    Ypts = source_tree |> root_elements
    for neartarget in nearlist(node)
        # near targets use direct evaluation
        I = index_range(neartarget)
        J = index_range(node)
        near_interaction!(C, K, Xpts, Ypts, B, I, J)
    end
    return C
end

"""
    near_interaction!(C,K,X,Y,σ,I,J)

Compute `C[i] <-- C[i] + ∑ⱼ K(X[i],Y[j])*σ[j]` for `i ∈ I`, `j ∈ J`. The default
implementation simply does a double loop over all `i ∈ I` and `j ∈ J`, but you
override this method for the type of your own custom kernel `K` if you have a
fast way of computing the full product in-place (e.g. using SIMD instructions).
"""
function near_interaction!(C, K, X, Y, σ, I, J)
    for i in I
        for j in J
            C[i] += K(X[i], Y[j]) * σ[j]
        end
    end
    return C
end


# generic fallback. May need to be overloaded for specific kernels.
"""
    centered_factor(K,x,Y::SourceTree)

Return a representative value of `K(x,y)` for `y ∈ Y`. This is used to help the
interpolation of `I(x) = ∑ᵢ K(x,yᵢ)` where `yᵢ ∈ Y` and points `x` in the far
field of `Y`. Defaults to `K(x,yc)` where `yc` is the center of `Y`.
"""
function centered_factor(K, x, Y)
    yc = center(Y)
    return K(x, yc)
end

"""
    inv_centered_factor(K,x,Y)

Inverse of [`centered_factor`](@ref).
"""
@inline function inv_centered_factor(K, x, Y)
    inv(centered_factor(K, x, Y))
end

# generic fallback. May need to be overloaded for specific kernels.
"""
    transfer_factor(K,x,Y::SourceTree)

Ratio between the centered factor of `Y` and the centered factor of its parent.
"""
function transfer_factor(K, x, Y)
    yparent = parent(Y)
    return inv_centered_factor(K, x, yparent) * centered_factor(K, x, Y)
end

function estimate_interpolation_order(kernel, partition, ds_func, tol, p)
    @timeit_debug "estimate interpolation order" begin
        nboxes = 20
        # pick N random nodes at each deepest level and estimate the interpolation
        # error by looking at the last chebyshev coefficients
        # for nodes in partition
        nodes = partition[end]
        n = length(nodes)
        Δn = n ÷ nboxes
        for i in 1:Δn:n
            node = nodes[i]
            er = Inf
            while er > tol
                N = ambient_dimension(node)
                ds = ds_func(node)
                ers = estimate_interpolation_error(kernel, node, ds, p)
                er, imax = findmax(ers)
                @debug ers, p
                if er > tol
                    p = ntuple(N) do i
                        i == imax ? p[i] + 1 : p[i]
                    end
                end
            end
        end
        # end
        order = p .- 1
    end
    return order
end

function estimate_interpolation_error(kernel, node::SourceTree{N,T}, ds, p) where {N,T}
    lb = SVector(sqrt(eps()), 0, -π)
    # lb     = SVector(sqrt(N)/(2N),0,-π)
    ub = SVector(sqrt(N) / N, π, π)
    domain = HyperRectangle(lb, ub)
    msh = UniformCartesianMesh(domain; step=ds)
    els = ElementIterator(msh)
    # pick a random cone
    I = CartesianIndex((1, 1, 1))
    rec = els[I]
    s = chebnodes(p, rec) # cheb points in parameter space
    x = map(si -> interp2cart(si, node), s)
    irange = 1:length(x)
    Tk = return_type(kernel, eltype(x), eltype(x))
    Tv = _density_type_from_kernel_type(Tk)
    vals = zeros(Tv, p...)
    Ypts = root_elements(node)[node.index_range]
    B = 2 * rand(Tv, length(Ypts)) .- 1
    jrange = 1:length(B)
    near_interaction!(vals, kernel, x, Ypts, B, irange, jrange)
    for i in eachindex(x)
        xi = x[i] # cartesian coordinate of node
        vals[i] = inv_centered_factor(kernel, xi, node) * vals[i]
    end
    coefs = _use_fftw() ? chebtransform_fftw!(reshape(vals, p...)) : chebtransform_native!(reshape(vals, p...))
    ntuple(i -> cheb_error_estimate(coefs, i), N)
end

function Base.show(io::IO, ifgf::IFGFOp)
    println(io, "IFGFOp of $(eltype(ifgf)) with range $(rowrange(ifgf)) × $(colrange(ifgf))")
    ctree = source_tree(ifgf)
    nodes = collect(AbstractTrees.PreOrderDFS(ctree))
    leaves = collect(AbstractTrees.Leaves(ctree))
    println(io, "\t number of nodes:                $(length(nodes))")
    println(io, "\t number of leaves:               $(length(leaves))")
    println(io, "\t number of active cones:         $(sum(x->length(active_cone_idxs(x)),nodes))")
    println(io, "\t maximum number of active cones: $(maximum(x->length(active_cone_idxs(x)),nodes))")
end
Base.show(io::IO, ::MIME"text/plain", ifgf::IFGFOp) = show(io, ifgf)
