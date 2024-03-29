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
and `i,j` entry given by `K(X[i],Y[j])`, where `K` is the underlying kernel
function, and `X` and `Y` are the elements of the `target_tree` and
`source_tree`, respectively.

## Fields:
- `kernel` : underlying kernel function
- `target_tree` : tree structure used for clustering the target points
- `source_tree` : tree structure used for clustering the source points
- `p::NTuple`: number of chebyshev poins per dimension used in the interpolation
  of far interactions
- `h::NTuple`: meshsize in parameter (per dimension) of the largest cones
"""
mutable struct IFGFOp{T} <: AbstractMatrix{T}
    const kernel
    const target_tree::TargetTree
    const source_tree::SourceTree
    const p::NTuple
    const h::NTuple
    const _source_tree_depth_partition
    mulbuffer::Union{Nothing,MulBuffer}
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

function Base.size(op::IFGFOp)
    sizeX = op |> target_tree |> root_elements |> length
    sizeY = op |> source_tree |> root_elements |> length
    return (sizeX, sizeY)
end

# disable getindex
function Base.getindex(::IFGFOp, args...)
    return error("`getindex(::IFGFOp,args...)` has been intentionally disabled")
end

"""
    assemble_ifgf(K, X, Y; tol, nmax=250)
    assemble_ifgf(K, X, Y; p, h, nmax=250)

Construct an approximation of the matrix `[K(x,y) for x ∈ X, y ∈ Y]` using the
[interpolated factored Green function method](https://arxiv.org/abs/2010.02857).
The parameter `tol` specifies the desired (relative) error in Frobenius norm of
the approximation, and `nmax` gives the maximum number of points in a leaf of
the dyadic trees used.

The kernel `K` must implement `wavenumber(::typeof(K)) --> Number` to return the
kernel's (typical) wavenumber; for non-oscillatory kernels this is zero.

When `tol` is passed, the number of interpolation points per dimension, `p`, and
the meshsize `h` is estimated based on some heuristic. If you know in advance a
good value of `p` and `h` for you function `K` and desired tolerance, it is
suggested you pass that as a keyword argument instead.
"""
function assemble_ifgf(
    kernel,
    target,
    source;
    tol = nothing,
    nmax = 250,
    p = nothing,
    # advanced parameters, intentionally not documented in the docstring since
    # their usage is subject to change
    η = nothing,
    h = nothing,
    splitter = DyadicSplitter(; nmax),
)
    U, V = eltype(target), eltype(source)
    Tk = return_type(kernel, U, V) # kernel type
    isconcretetype(Tk) || error("unable to infer a concrete type for the kernel")
    N, Td = length(U), eltype(U)
    # 2. figure out if tolerance + h + p make sense, correct if possible
    h = isa(h, Number) ? ntuple(i -> Td(h), N) : h
    p = isa(p, Number) ? ntuple(i -> p, N) : p
    if isnothing(tol)
        (isnothing(p) || isnothing(h)) &&
            error("you must pass either tol or p and h as keyword arguments")
    else
        isnothing(h) || (h = nothing; @warn "ignoring h since tol was passed")
    end
    # 3. Create an admissibility condition unless one is passed
    η = isnothing(η) ? √N : η
    adm = AdmissibilityCondition(η)
    # Trees creation. Note that we must pass the type of data that will be
    # stored in each node of the cluster tree.
    Xtree = ClusterTree{TargetTreeData}(target, splitter)
    if target === source
        # create a shadow tree with the same points/boxes but with data
        Ytree = shadow_tree(Xtree, SourceTreeData{N,Td,typeof(Xtree)})
    else
        Ytree = ClusterTree{SourceTreeData{N,Td,typeof(Xtree)}}(source, splitter)
    end
    partition = partition_by_depth(Ytree)
    # Create a reference domain inside which all nodes must lie.
    smin = 1e-10
    smax = 1 / η
    interp_domain = if N == 2
        HyperRectangle{N,Td}((smin, -π), (smax, π))
    elseif N == 3
        HyperRectangle{N,Td}((smin, 0, -π), (smax, π, π))
    end
    # We need the the function that computes a cone meshsize given a node, and
    # the number of interpolation points per dimension. In the case a tolerance
    # was passed, that takes priority and we try to find some "good" candidates
    # for `p` and `h`. We apply a simple heuristic for choosing `p`, and an
    # adaptive procedure for chosing `h` after, whereas we estimate the
    # interpolaton order by looking at the last chebyshev coefficients in a few
    # of the nodes of the source tree
    k = wavenumber(kernel)
    p = if isnothing(p)
        if tol > 1e-3
            ntuple(i -> 4, N)
        elseif tol > 1e-12
            ntuple(i -> 8, N)
        else
            ntuple(i -> 16, N)
        end
    else
        p
    end
    if !isnothing(tol)
        h = estimate_cone_size(partition, p, kernel, tol, Tk, interp_domain)
    end
    cone_size = ConeDomainSize(k, h)
    @debug "---Cones meshsize           = " h
    @debug "---Number of interp. points = " p
    ## Interaction list.
    # The parameter `nmin` is the minimum number of points below which all
    # interactions are considered near
    nmin = prod(p)
    _compute_interaction_list!(Ytree, Xtree, adm, nmin)
    ## Cone lists
    # Finally compute the active cones.
    _compute_cone_list!(partition, Val(p), cone_size, interp_domain)
    # call main constructor
    ifgf = IFGFOp{Tk}(kernel, Xtree, Ytree, p, promote(h...), partition, nothing)
    return ifgf
end

"""
    _compute_interaction_list!(source, target, adm)

For each node in `source`, compute its `nearlist` and `farlist`. The `nearlist`
contains all nodes in `target` for which the interaction must be computed using
the direct summation.  The `farlist` includes all nodes in `target_tree` for
which an accurate interpolant in available in the source node.

The `adm` function is called through `adm(target,source)` to determine if
`target` is in the `farlist` of source. When that is not the case we recurse on
their children unless both are a leaves, in which case `target` is added to the
`nearlist` of source.
"""
@noinline function _compute_interaction_list!(
    source::SourceTree,
    target::TargetTree,
    adm,
    nmin,
)
    # if either box is empty return immediately
    if length(source) == 0 || length(target) == 0
        return nothing
    end
    # handle the various possibilities
    if isleaf(source) && length(source) < nmin # small leafs do not interpolate
        _addtonearlist!(source, target)
    elseif adm(target, source)
        _addtofarlist!(source, target)
    elseif isleaf(target) && isleaf(source)
        _addtonearlist!(source, target)
    elseif isleaf(target)
        # recurse on children of source
        for source_child in children(source)
            _compute_interaction_list!(source_child, target, adm, nmin)
        end
    elseif isleaf(source)
        # recurse on children of target
        for target_child in children(target)
            _compute_interaction_list!(source, target_child, adm, nmin)
        end
    else
        # TODO: when both have children, how to recurse down? Two strategies
        # below:
        # strategy 1: recurse on children of largest box
        if radius(source) > radius(target)
            for source_child in children(source)
                _compute_interaction_list!(source_child, target, adm, nmin)
            end
        else
            for target_child in children(target)
                _compute_interaction_list!(source, target_child, adm, nmin)
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
    _compute_cone_list!(partition, p, ds_func)

For each node in `partition`, compute the cones which are necessary to cover
both the target points on its `farlist` as well as the interpolation points on
its parent's cones. `p` denotes the number of interpolation points per
dimension, and `ds_func(node)` gives the meshsize of the domain in
interpolation-space for `node`.
"""
@noinline function _compute_cone_list!(partition, p, ds_func, domain_)
    T = float_type(domain_)
    refnodes = chebnodes(p, T)
    for depth in partition
        @usethreads for source in depth
            length(source) == 0 && continue # skip empty nodes
            ds = ds_func(source)
            domain = _use_minimal_conedomain() ? interpolation_domain(source, p) : domain_
            all(low_corner(domain) .< high_corner(domain)) || continue
            _init_msh!(source, domain, ds)
            lck = _usethreads() ? ReentrantLock() : nothing
            # create all cones needed to cover interpolation points of parents
            if !isroot(source)
                source_parent = parent(source)
                @usethreads for I in active_cone_idxs(source_parent)
                    rec = cone_domain(source_parent, I)
                    lc, hc = low_corner(rec), high_corner(rec)
                    c0 = (lc + hc) / 2
                    c1 = (hc - lc) / 2
                    # cheb = cheb2nodes(p,rec)
                    for ŝ in refnodes # interpolation node in interpolation space
                        s = c0 + c1 .* ŝ
                        # interpolation node in physical space
                        x = interp2cart(s, source_parent)
                        # mesh index for interpolation point
                        I, _ = cone_index(x, source)
                        # initialize cone if needed
                        _addnewcone!(source, I, lck)
                    end
                    # end
                end
            end
            # initialize all cones needed to cover far field.
            @usethreads for far_target in farlist(source)
                @usethreads for x in elements(far_target) # target points
                    I, _ = cone_index(x, source)
                    _addnewcone!(source, I, lck)
                end
            end
            sort!(conedatadict(source))
        end
    end
    return partition
end
###############################################################################
#                          Linear Algebra Functions                           #
###############################################################################

"""
    mul!(y,A::IFGFOp,x,a,b;global_index=true)

Setting `global_index=false` uses the internal indexing system of the `IFGFOp`,
and therefore does not permute the vectors `x` and `y` to perform the
multiplication.
"""
function LinearAlgebra.mul!(
    y::AbstractVector,
    A::IFGFOp,
    x::AbstractVector,
    a::Number,
    b::Number;
    global_index = true,
)
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
    Ty = Base.promote_op(*, TA, Tx) # interpolant type (same as output type)
    if !isa(A.mulbuffer, MulBuffer{Ty})
        @debug "creating buffer of type $Ty to perform multiplication"
        bufs = _allocate_buffers!(partition, prod(A.p), Ty)
        # create an fftw plan for the chebtransform. Since `Ti` may be an `SVector`, we plan on
        # its element type and perform the FFT on each component
        plan = FFTW.plan_r2r!(
            zeros(eltype(Ty), A.p),
            FFTW.REDFT00;
            flags = FFTW.PATIENT | FFTW.UNALIGNED,
        )
        # update buffer field
        A.mulbuffer = MulBuffer(bufs[1], bufs[2], plan)
    end
    rtree = target_tree(A) # row tree
    ctree = source_tree(A) # column tree
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
    _gemv!(y, K, rtree, partition, x, Val(A.p), A.mulbuffer)
    # permute output
    global_index && permute!(y, glob2loc(rtree))
    return y
end

function LinearAlgebra.mul!(
    y::AbstractMatrix,
    A::IFGFOp,
    x::AbstractMatrix,
    a::Number,
    b::Number;
    global_index = true,
)
    ncol = size(x, 2)
    for i in 1:ncol
        mul!(view(y, :, i), A, view(x, :, i), a, b; global_index)
    end
    return y
end

# memory need for forward map
function _allocate_buffers!(partition, p, Tv)
    nmax = 0
    for nodes in partition
        istart = 0
        iend = 0
        for node in nodes
            for I in active_cone_idxs(node)
                istart = iend + 1
                iend = istart + p - 1
                node.data.conedatadict[I] = istart:iend
            end
        end
        nmax = max(nmax, iend)
    end
    buff1 = Vector{Tv}(undef, nmax)
    buff2 = Vector{Tv}(undef, nmax)
    return (buff1, buff2)
end

@noinline function _gemv!(C, K, target_tree, partition, B, p::Val{P}, mulbuffer) where {P}
    coefs, next_coefs = mulbuffer.current_coefs, mulbuffer.next_coefs
    plan = mulbuffer.plan
    nt = Threads.nthreads()
    # Cthreads = [zero(C) for _ in 1:nt]
    chn = Channel{typeof(C)}(nt)
    foreach(i -> put!(chn, zero(C)), 1:nt)
    dmax = length(partition)
    for d in dmax:-1:1
        nodes = partition[d]
        fill!(next_coefs, zero(eltype(next_coefs)))
        @usethreads for node in nodes
            Cloc = take!(chn)
            length(node) == 0 && continue
            if isleaf(node)
                _particles_to_particles!(Cloc, K, target_tree, node, B, node)
                _particles_to_interpolants!(node, K, B, p, next_coefs)
            else
                _interpolants_to_interpolants(node, K, p, next_coefs, coefs)
            end
            _chebtransform!(node, p, plan, next_coefs)
            _interpolants_to_particles!(Cloc, K, target_tree, node, p, next_coefs)
            put!(chn, Cloc)
        end
        coefs, next_coefs = next_coefs, coefs
    end
    # reduction stage
    for _ in 1:nt
        Cloc = take!(chn)
        axpy!(1, Cloc, C)
    end
    return C
end

function _interpolants_to_interpolants(node, K, p::Val{P}, buff, cbuff) where {P}
    N = length(P)
    T = float_type(node)
    refnodes = chebnodes(p, T)
    @usethreads for I in active_cone_idxs(node)
        vals = reshape(view(buff, conedata(node, I)), P)
        rec = cone_domain(node, I)
        lc, hc = low_corner(rec), high_corner(rec)
        c0 = (lc + hc) / 2
        c1 = (hc - lc) / 2
        for chd in children(node)
            length(chd) == 0 && continue
            for (i, ŝ) in enumerate(refnodes)
                s = c0 + c1 .* ŝ
                x = interp2cart(s, node) # cheb point in physical space
                idxcone, si = cone_index(x, chd)
                # transfer block's interpolant to parent's interpolants
                rec_chd = cone_domain(chd, idxcone)
                coefs   = view(cbuff, conedata(chd, idxcone))
                vals[i] += transfer_factor(K, x, chd) * chebeval(coefs, si, rec_chd, p)
            end
        end
    end
    return node
end

# transfer the data from a left cell to its right neighbors
function _transfer_right(I, vals_left, buff, dict, ::Val{P}) where {P}
    N = length(P)
    for dim in 1:N
        Ir = increment_index(I, dim)
        if haskey(dict, Ir)
            vals_right = reshape(view(buff, dict[Ir]), P)
            # HACK: because the line below allocates (don't know why), we have
            # to build the indices manually to copy the data.
            # copy!(selectdim(vals_right,dim,P[dim]),selectdim(vals,dim,1))
            idxs_left = ntuple(i -> i == dim ? (P[i]:P[i]) : 1:P[i], N) |> CartesianIndices
            idxs_right = ntuple(i -> i == dim ? (1:1) : 1:P[i], N) |> CartesianIndices
            for i in 1:length(idxs_left)
                vals_right[idxs_left[i]] = vals_left[idxs_right[i]]
            end
        end
    end
end

function _chebtransform!(node::SourceTree, ::Val{P}, plan, buff) where {P}
    @usethreads for I in active_cone_idxs(node)
        vals = view(buff, conedata(node, I))
        # transform vals to coefs in place. Coefs will be used later when the
        # current node transfers its expansion to its parent
        coefs = if _use_fftw()
            chebtransform_fftw!(reshape(vals, P...), plan)
        else
            chebtransform_native!(reshape(vals, P...))
        end
    end
    return node
end

function _particles_to_interpolants!(node::SourceTree, K, B, p::Val, buff)
    Ypts = node |> root_elements
    for I in active_cone_idxs(node)
        rec = cone_domain(node, I)
        s = chebnodes(p, rec) # cheb points in parameter space
        x = map(si -> interp2cart(si, node), s) |> vec
        irange = 1:length(x)
        jrange = index_range(node)
        vals = view(buff, conedata(node, I))
        near_interaction!(vals, K, x, Ypts, B, irange, jrange)
        for i in eachindex(x)
            xi = x[i] # cartesian coordinate of node
            vals[i] = inv_centered_factor(K, xi, node) * vals[i]
        end
    end
    return node
end

function _interpolants_to_particles!(
    C,
    K,
    target_tree,
    node::SourceTree,
    p::Val{P},
    buff,
) where {P}
    Xpts = target_tree |> root_elements
    @usethreads for far_target in farlist(node)
        @usethreads for i in index_range(far_target)
            x = Xpts[i]
            I, s = cone_index(coords(x), node)
            rec = cone_domain(node, I)
            coefs = view(buff, conedata(node, I))
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
implementation simply does a double loop over all `i ∈ I` and `j ∈ J`: override
this method for the type of your own custom kernel `K` if you have a fast way of
computing the full product in-place (e.g. using SIMD instructions).
"""
function near_interaction!(args...)
    @warn "using generic fallback for near_interaction" maxlog = 1
    return near_interaction_naive!(args...)
end

function near_interaction_naive!(C, K, X, Y, σ, I, J)
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

Return a representative scalar value of `K(x,y)` for `y ∈ Y`. This is used to
help the interpolation of `I(x) = ∑ᵢ K(x,yᵢ)` where `yᵢ ∈ Y` and points `x` in
the far field of `Y`. Defaults to `K(x,yc)` where `yc` is the center of `Y`.
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
    return inv(centered_factor(K, x, Y))
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

function estimate_cone_size(partition, p, kernel, tol, Tk, interp_domain)
    N = ambient_dimension(interp_domain)
    ntrials = 10
    h = ntuple(i -> π + 1e-4, N)
    iter = 1
    k = wavenumber(kernel)
    while true
        func = ConeDomainSize(k, h)
        acc = svector(i -> 0.0, N)
        for i in 1:ntrials
            node = partition[end] |> rand # random leaf
            ds   = func(node)
            msh  = UniformCartesianMesh(interp_domain; step = ds)
            acc  += estimate_interpolation_error(kernel, node, msh, p, Tk)
        end
        ers = acc ./ ntrials
        imax = argmax(ers)
        if maximum(ers) > tol
            @debug h, ers
            iter += 1 # going into next iteration, refine the mesh
            h = h ./ iter
        else
            @debug "Parameters: p=$p, h=$h, ers=$ers"
            break
        end
    end
    return h
end

function estimate_interpolation_error(kernel, node::SourceTree, msh, p, Tk)
    N = ambient_dimension(msh)
    ncones = 10
    acc = svector(i -> 0.0, N)
    for _ in 1:ncones
        # pick a random cone
        I = rand(CartesianIndices(msh))
        rec = msh[I]
        s = chebnodes(p, rec) # cheb points in parameter space
        x = map(si -> interp2cart(si, node), s)
        irange = 1:length(x)
        Ypts = root_elements(node)[node.index_range]
        TB = Base.promote_op(pinv, Tk)
        B = rand(TB, length(Ypts))
        vals = zeros(Base.promote_op(*, Tk, eltype(B)), p...)
        jrange = 1:length(B)
        near_interaction_naive!(vals, kernel, x, Ypts, B, irange, jrange)
        for i in eachindex(x)
            xi = x[i] # cartesian coordinate of node
            vals[i] = inv_centered_factor(kernel, xi, node) * vals[i]
        end
        coefs = if _use_fftw()
            chebtransform_fftw!(reshape(vals, p...))
        else
            chebtransform_native!(reshape(vals, p...))
        end
        acc += svector(i -> cheb_error_estimate(coefs, i), N)
    end
    ers = acc ./ ncones
    return ers
end

function Base.show(io::IO, ifgf::IFGFOp)
    println(
        io,
        "IFGFOp of $(eltype(ifgf)) with range $(rowrange(ifgf)) × $(colrange(ifgf))",
    )
    ctree = source_tree(ifgf)
    nodes = collect(AbstractTrees.PreOrderDFS(ctree))
    leaves = collect(AbstractTrees.Leaves(ctree))
    println(io, "|-- interpolation points per cone: $(ifgf.p)")
    println(io, "|-- maximal meshsize per cone:     $(trunc.(ifgf.h,sigdigits=3))")
    println(io, "|-- number of nodes:               $(length(nodes))")
    println(io, "|-- number of leaves:              $(length(leaves))")
    println(io, "|-- number of active cones:        $(num_active_cones(ifgf))")
    return io
end
Base.show(io::IO, ::MIME"text/plain", ifgf::IFGFOp) = show(io, ifgf)

function num_active_cones(ifgf::IFGFOp)
    ctree = source_tree(ifgf)
    nodes = collect(AbstractTrees.PreOrderDFS(ctree))
    return sum(x -> length(active_cone_idxs(x)), nodes)
end
