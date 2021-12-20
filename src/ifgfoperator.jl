"""
    struct IFGFOp{T} <: AbstractMatrix{T}
"""
struct IFGFOp{T} <: AbstractMatrix{T}
    kernel
    target_tree::TargetTree
    source_tree::Union{SourceTree,SourceTreeLite}
    p::NTuple
    ds_func::Function
    _source_tree_depth_partition
end

islite(ifgf::IFGFOp) = source_tree(ifgf) isa SourceTreeLite

# getters
rowrange(ifgf::IFGFOp)        = Trees.index_range(ifgf.target_tree)
colrange(ifgf::IFGFOp)        = Trees.index_range(ifgf.source_tree)
kernel(op::IFGFOp)            = op.kernel
target_tree(op::IFGFOp)       = op.target_tree
source_tree(op::IFGFOp)       = op.source_tree
polynomials_order(op::IFGFOp) = op.p
meshsize_func(op::IFGFOp)     = op.ds_func

function Base.size(op::IFGFOp)
    sizeX = op |> target_tree |> root_elements |> length
    sizeY = op |> source_tree |> root_elements |> length
    return (sizeX,sizeY)
end

# disable getindex
function Base.getindex(::IFGFOp,args...)
    error()
end

"""
    IFGFOperator(kernel,Ypoints,Xpoints;splitter,p,ds_func,lite=true,threads=true)

Create the source and target tree-structures for clustering the source `Ypoints`
and target `Xpoints` using the splitting strategy of `splitter`. Both trees are
initialized (i.e. the interaction list and the cone list are computed).

# Arguments
- `kernel`: kernel of the IFGF operator.
- `Ypoints`: vector of source points.
- `Xpoints`: vector of observation points.
- `splitter`: splitting strategy.
- `p`: the number of interpolation points
            per dimension in interpolation coordinates. In 3D, `p = (ns,nθ,nϕ)`.
            In 2D, `p = (ns,nθ)`.
- `ds_func`: function `ds_func(source_tree) = ds` that returns the size of the cone domains
             per dimension in interpolation coordinates. In 3D, `ds = (dss,dsθ,dsϕ)`.
             In 2D, `ds = (dss,dsθ)`.
"""
function IFGFOp(kernel,Xpoints,Ypoints;splitter,ds_func,adm=modified_admissible_condition,lite=true,threads=true,tol=0,p=nothing)
    # infer the type of the matrix, then compute the type of density to be interpolated
    U,V = eltype(Xpoints), eltype(Ypoints)
    T = hasmethod(return_type,Tuple{typeof(kernel)}) ? return_type(kernel) : Base.promote_op(kernel,U,V)
    assert_concrete_type(T)
    Tv = _density_type_from_kernel_type(T)
    # create source and target trees
    @timeit_debug "source tree initialization" begin
        source_tree = initialize_source_tree(;Ypoints,splitter,Vdatatype=Tv,lite)
    end
    @timeit_debug "target tree initialization" begin
        target_tree = initialize_target_tree(;Xpoints,splitter)
    end
    partition = Trees.partition_by_depth(source_tree)

    if p === nothing
        @assert tol > 0
        p = estimate_interpolation_order(kernel,partition,ds_func,tol,(2,2,2))
        @info p
    end
    # intialize ifgf object
    ifgf = IFGFOp{T}(kernel, target_tree, source_tree, p, ds_func, partition)
    # update interaction lists
    @timeit_debug "cell interactions" begin
        compute_interaction_list!(ifgf,adm)
    end
    @timeit_debug "cone lists" begin
        compute_cone_list!(ifgf,threads)
    end
    return ifgf
end

function estimate_interpolation_order(kernel,partition,ds_func,tol,p)
    nboxes = 20
    # pick N random nodes at each deepest level and estimate the interpolation
    # error by looking at the last chebyshev coefficients
    # for nodes in partition
    nodes = partition[end]
    n  = length(nodes)
    Δn = n ÷ nboxes
    for i in 1:Δn:n
        node = nodes[i]
        er = Inf
        while er > tol
            N = ambient_dimension(node)
            ds   = ds_func(node)
            ers   = estimate_interpolation_error(kernel,node,ds,p)
            er,imax = findmax(ers)
            @debug ers,p
            if er > tol
                p = ntuple(N) do i
                    i == imax ? p[i]+1 : p[i]
                end
            end
        end
    end
    # end
    return p
end

function estimate_interpolation_error(kernel,node::SourceTreeLite{N,T,V},ds,p) where {N,T,V}
    # lb     = SVector(sqrt(eps()),0,-π)
    lb     = SVector(sqrt(N)/(2N),0,-π)
    ub     = SVector(sqrt(N)/N,π,π)
    domain = HyperRectangle(lb,ub)
    msh    = UniformCartesianMesh(domain; step = ds)
    els    = ElementIterator(msh)
    # pick a random cone
    I      = CartesianIndex((1,1,1))
    rec    = els[I]
    s      = cheb2nodes(p,rec) # cheb points in parameter space
    x      = map(si->interp2cart(si,node),s)
    irange = 1:length(x)
    vals   = zeros(V,p...)
    Ypts   = root_elements(node)[node.index_range]
    B      = 2*rand(V,length(Ypts)) .- 1
    jrange = 1:length(B)
    near_interaction!(vals,kernel,x,Ypts,B,irange,jrange)
    for i in eachindex(x)
        xi = x[i] # cartesian coordinate of node
        vals[i] /= centered_factor(kernel,xi,node)
    end
    coefs = chebcoefs!(reshape(vals,p...))
    ntuple(i->cheb_error_estimate(coefs,i),N)
end

"""
    compute_interaction_list!(ifgf::IFGFOperator,adm)

For each node in `source_tree(ifgf)`, compute its `nearlist` and `farlist`. The
`nearlist` contains all nodes in `target_tree(ifgf)` for which the interaction must be
computed using the direct summation.  The `farlist` includes all nodes in
`target_tree` for which an accurate interpolant in available in the source node.

The `adm` function is called through `adm(target,source)` to determine if
`target` is in the `farlist` of source. When that is not the case we recurse
on their children unless both are a leaves, in which
case `target` is added to the `nearlist` of source.
"""
function compute_interaction_list!(ifgf::IFGFOp,adm)
    _compute_interaction_list!(source_tree(ifgf),target_tree(ifgf),adm)
    return ifgf
end

function _compute_interaction_list!(source::Union{SourceTree,SourceTreeLite}, target::TargetTree, adm)
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
        # below: recurse on largest one, or recurse on both.
        # strategy 1: recurse on children of largest box
        # if radius(source) > radius(target)
        #     for source_child in children(source)
        #         compute_interaction_list!(source_child, target, adm)
        #     end
        # else
        #     for target_child in children(target)
        #         compute_interaction_list!(source, target_child, adm)
        #     end
        # end
        # strategy 2: recurse on childre of both target and source
        for source_child in children(source)
            for target_child in children(target)
                _compute_interaction_list!(source_child, target_child, adm)
            end
        end
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
function compute_cone_list!(ifgf::IFGFOp,threads)
    p    = polynomials_order(ifgf)
    func = meshsize_func(ifgf)
    X    = target_tree(ifgf) |> root_elements
    partition = ifgf._source_tree_depth_partition
    _compute_cone_list!(partition,X,Val(p),func,threads)
    return ifgf
end

function _compute_cone_list!(partition,X,p,func,threads)
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

function _initialize_cone_interpolants!(source::SourceTree, X, p::Val{P}, ds_func) where {P}
    length(source) == 0 && (return source)
    ds = ds_func(source)
    # find minimal axis-aligned bounding box for interpolation points
    domain,farpcoords,parpcoords = compute_interpolation_domain(source, p)
    all(low_corner(domain) .< high_corner(domain)) || (return source)
    _init_msh!(source, domain, ds)
    # if not a root, also create all cones needed to cover interpolation points
    # of parents
    if !isroot(source)
        source_parent = parent(source)
        ipts = source_parent.data.ipts
        for (i, s) in enumerate(parpcoords)
            # idxcone, s = cone_index(x, source)
            idxcone = element_index(s,msh(source))
            _addnewcone!(source, idxcone, p)
            _addidxtoparlist!(source, idxcone, i, s)
        end
    end
    # initialize all cones needed to cover far field.
    cc = 0
    for far_target in farlist(source)
        for i in index_range(far_target) # target points
            cc += 1
            # idxcone, s = cone_index(center(X[i]), source)
            s = farpcoords[cc]
            idxcone = element_index(s,msh(source))
            _addnewcone!(source, idxcone, p)
            _addidxtofarlist!(source, idxcone, i, s)
        end
    end
    allocate_ivals!(source, p)
    return source
end

function _initialize_cone_interpolants!(source::SourceTreeLite{N}, X, p::Val{P}, ds_func) where {N,P}
    length(source) == 0 && (return source)
    Iold = zero(CartesianIndex{N})
    ds = ds_func(source)
    # find minimal axis-aligned bounding box for interpolation points
    domain         = compute_interpolation_domain(source,p)
    all(low_corner(domain) .< high_corner(domain)) || (return source)
    _init_msh!(source, domain, ds)
    # if not a root, also create all cones needed to cover interpolation points
    # of parents
    refnodes = cheb2nodes(p)
    if !isroot(source)
        source_parent = parent(source)
        for I in active_cone_idxs(source_parent)
            rec  = cone_domain(source_parent,I)
            lc,hc = low_corner(rec), high_corner(rec)
            c0 = (lc+hc)/2
            c1 = (hc-lc)/2
            cheb  = map(x-> c0 + c1 .* x,refnodes)
            # cheb = cheb2nodes(p,rec)
            for si in cheb # interpolation node in interpolation space
                # interpolation node in physical space
                xi = interp2cart(si,source_parent)
                # mesh index for interpolation point
                Inew,_ = cone_index(xi,source)
                Iold === Inew && continue
                # initialize cone if needed
                _addnewcone!(source,Inew,p)
                Iold = Inew
            end
        end
    end
    # initialize all cones needed to cover far field.
    for far_target in farlist(source)
        for x in Trees.elements(far_target) # target points
            Inew,_ = cone_index(x,source)
            Iold === Inew && continue
            _addnewcone!(source,Inew,p)
            Iold = Inew
        end
    end
    return source
end

function LinearAlgebra.mul!(y::AbstractVector, A::IFGFOp, x::AbstractVector, a::Number, b::Number;global_index=true,threads=false)
    # since the IFGF represents A = Pr*K*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end.
    T     = eltype(A)
    rtree = target_tree(A)
    ctree = source_tree(A)
    # permute input
    if global_index
        x         = x[ctree.loc2glob]
        y         = permute!(y,rtree.loc2glob)
        rmul!(x,a) # multiply in place since this is a new copy, so does not mutate exterior x
    else
        x = a*x # new copy of x
    end
    iszero(b) ? fill!(y,zero(eltype(y))) : rmul!(y,b)
    # extract the kernel and place a function barrier for heavy computation
    K     = kernel(A)
    partition = A._source_tree_depth_partition
    if threads
        _gemv_threaded!(y,K,rtree,partition,x,T,Val(A.p))
    else
        _gemv!(y,K,rtree,partition,x,T,Val(A.p))
    end
    # permute output
    global_index && invpermute!(y,loc2glob(rtree))
    return y
end

function _gemv!(C,K,target_tree,partition,B,T,p::Val{P}) where {P}
    Tv = _density_type_from_kernel_type(T)
    # do some planning for the fft
    plan = FFTW.plan_r2r!(zeros(Tv,P),FFTW.REDFT00;flags=FFTW.PATIENT)
    for depth in Iterators.reverse(partition)
        for node in depth
            length(node) == 0 && continue
            # allocate necessary interpolation data and perform necessary
            # precomputations
            if isleaf(node)
                @timeit_debug "P2M" _particle_to_moments!(node,K,B,p,plan)
            else
                @timeit_debug "M2M" _moments_to_moments!(node,K,B,p,plan)
            end
            @timeit_debug "M2P" _moments_to_particles!(C,K,target_tree,node,p)
            @timeit_debug "P2P" _particles_particles!(C,K,target_tree,node,B,node)
        end
    end
    # since root has not parent, it must free its own ivals
    root = partition |> first |> first
    free_ivals!(root)
end

function _gemv_threaded!(C,K,target_tree,partition,B,T,p::Val{P}) where {P}
    Tv = _density_type_from_kernel_type(T)
    # do some planning for the fft
    plan = FFTW.plan_r2r!(zeros(Tv,P),FFTW.REDFT00;flags=FFTW.PATIENT)
    # the threading strategy is to create `nt` copies of the output vector, and
    # for each depth thread all nodes on that depth with the local version of
    # the output vector.
    nt = Threads.nthreads()
    Cthreads = [zero(C) for _ in 1:nt]
    for depth in Iterators.reverse(partition)
        Threads.@threads for node in depth
            length(node) == 0 && continue
            id = Threads.threadid()
            if isleaf(node)
                _particle_to_moments!(node,K,B,p,plan)
            else
                _moments_to_moments!(node,K,B,p,plan)
            end
            _moments_to_particles!(Cthreads[id],K,target_tree,node,p)
            _particles_particles!(Cthreads[id],K,target_tree,node,B,node)
        end
    end
    # reduction stage
    for i in 1:nt
        axpy!(1,Cthreads[i],C)
    end
    # since root has not parent, it must free its own ivals
    root = partition |> first |> first
    free_ivals!(root)
end

function _moments_to_moments!(node::SourceTree{N,Td,Tv},K,B,p::Val{P},plan) where {N,Td,Tv,P}
    isleaf(node) && return _particle_to_moments!(node,K,B,p,plan)
    X     = node.data.ipts
    ivals = node.data.ivals
    fill!(ivals,zero(eltype(ivals)))
    for chd in children(node)
        length(chd) == 0 && continue
        for (I,data) in conedatadict(chd)
            coefs     = getivals(chd,I)
            pcoords   = parpcoords(chd,I)
            idxs   = data.paridxs
            rec    = cone_domain(chd,I)
            for (n,i) in enumerate(idxs)
                si = pcoords[n]
                ivals[i]  += chebeval(coefs,si,rec,p) * transfer_factor(K,X[i],chd)
            end
        end
    end
    # transform from vals to cheb coefficients
    L = prod(P)
    kmax = length(ivals)/L |> Int
    # TODO: it should be possible to do a batch fft instead of several
    # individual ones.
    for k in 1:kmax
        vals = @views ivals[(k-1)*L+1:k*L]
        chebcoefs!(reshape(vals,P...),plan)
    end
    return node
end

function _moments_to_moments!(node::SourceTreeLite{N,Td,Tv},K,B,p::Val{P},plan) where {N,Td,Tv,P}
    allocate_ivals!(node,p)
    for I in active_cone_idxs(node)
        rec = cone_domain(node,I)
        s          = cheb2nodes(p,rec) # cheb points in parameter space
        x          = map(si->interp2cart(si,node),s) # cheb points in physical space
        vals       = getivals(node,I)
        for chd in children(node)
            length(chd) == 0 && continue
            idxold = zero(CartesianIndex{N})
            for i in eachindex(x)
                idxnew,si   = cone_index(x[i], chd)
                if idxnew !== idxold # => switch interpolant
                    # transfer block's interpolant to parent's interpolants
                    rec   = cone_domains(chd)[idxnew]
                    coefs = getivals(chd,idxnew)
                    idxold = idxnew
                end
                vals[i] += chebeval(coefs,si,rec,p) * transfer_factor(K,x[i],chd)
            end
        end
        coefs = chebcoefs!(reshape(vals,P...),plan)
        # er    = cheb_error_estimate(coefs,1),cheb_error_estimate(coefs,2),cheb_error_estimate(coefs,3)
        # @info er
    end
    # interps on children no longer needed
    for chd in children(node)
        free_ivals!(chd)
    end
    return node
end

function _particle_to_moments!(node::SourceTree{N,Td,Tv},K,B,p::Val{P},plan) where {N,Td,Tv,P}
    Ypts = node |> root_elements
    X    = node.data.ipts
    out  = node.data.ivals
    fill!(out,zero(eltype(out)))
    idxs = 1:length(X)
    jrange = index_range(node)
    # leaf nodes compute their own interp vals
    near_interaction!(out,K,X,Ypts,B,idxs,jrange)
    for i in idxs
        xi = X[i] # cartesian coordinate of node
        out[i] /= centered_factor(K,xi,node)
    end
    # transform from vals to cheb coefficients
    L = prod(P)
    kmax = length(out)/L |> Int
    # TODO: it should be possible to do a batch fft instead of several
    # individual ones.
    for k in 1:kmax
        vals = @views out[(k-1)*L+1:k*L]
        chebcoefs!(reshape(vals,P...),plan)
    end
    return node
end

function _particle_to_moments!(node::SourceTreeLite{N,Td,Tv},K,B,p::Val{P},plan) where {N,Td,Tv,P}
    Ypts = node |> root_elements
    allocate_ivals!(node,p)
    for I in active_cone_idxs(node)
        rec = cone_domain(node,I)
        s          = cheb2nodes(p,rec) # cheb points in parameter space
        x          = map(si->interp2cart(si,node),s)
        irange = 1:length(x)
        jrange = index_range(node)
        vals       = getivals(node,I)
        near_interaction!(vals,K,x,Ypts,B,irange,jrange)
        for i in eachindex(x)
            xi = x[i] # cartesian coordinate of node
            vals[i] /= centered_factor(K,xi,node)
        end
        coefs = chebcoefs!(reshape(vals,P...),plan)
    end
    return node
end

function _moments_to_particles!(C,K,target_tree,node::SourceTree,p::Val{P}) where {P}
    Xpts  = target_tree |> root_elements
    for (I,data) in conedatadict(node)
        idxs    = data.faridxs
        rec     = cone_domain(node,I)
        coefs   = getivals(node,I)
        pcoords = farpcoords(node,I)
        for (n,i) in enumerate(idxs)
            s = pcoords[n]
            C[i]  += chebeval(coefs,s,rec,p) * centered_factor(K,Xpts[i],node)
        end
    end
    return nothing
end

function _moments_to_particles!(C,K,target_tree,node::SourceTreeLite{N,T},p::Val{P}) where {N,T,P}
    Xpts   = target_tree |> root_elements
    ivals  = node.data.ivals
    isempty(ivals) && (return nothing)
    # initialize some variables outside the for loop that will be used to
    # determine if we should switch interpolant or not. This is a performance
    # optimization since most of the time consecutive poins in the far field
    # will be in the same cone interpolant.
    Iold   = zero(CartesianIndex{N})
    rec    = HyperRectangle{N,T}(ntuple(i->0,N),ntuple(i->0,N))
    coefs  = @view ivals[1:1]
    for far_target in farlist(node)
        for i in index_range(far_target)
            x         = Xpts[i]
            Inew,s    = cone_index(coords(x),node)
            if Inew !== Iold
                idxrange      = coneidxrange(node,Inew)
                rec           = cone_domain(node,Inew)
                coefs  = @view ivals[idxrange]
                Iold          = Inew
            end
            C[i]  += chebeval(coefs,s,rec,p) * centered_factor(K,Xpts[i],node)
        end
    end
    return nothing
end

function _particles_particles!(C,K,target_tree,source_tree,B,node)
    Xpts = target_tree |> root_elements
    Ypts = source_tree |> root_elements
    for neartarget in nearlist(node)
        # near targets use direct evaluation
        I = index_range(neartarget)
        J = index_range(node)
        near_interaction!(C,K,Xpts,Ypts,B,I,J)
   end
    return nothing
end

"""
    near_interaction!(C,K,X,Y,σ,I,J)

Compute `C[i] <-- C[i] + ∑ⱼ K(X[i],Y[j])*σ[j]` for `i ∈ I`, `j ∈ J`.
"""
function near_interaction!(C,K,X,Y,σ,I,J)
    for i in I
        for j in J
            C[i] += K(X[i], Y[j]) * σ[j]
        end
    end
end


# generic fallback. May need to be overloaded for specific kernels.
function centered_factor(K,x,ysource::Union{SourceTree,SourceTreeLite})
    yc = center(ysource)
    return K(x,yc)
end

# generic fallback. May need to be overloaded for specific kernels.
function transfer_factor(K,x,ysource::Union{SourceTree,SourceTreeLite})
    yparent = parent(ysource)
    return centered_factor(K,x,ysource)/centered_factor(K,x,yparent)
end

function Base.show(io::IO,ifgf::IFGFOp)
    println(io,"IFGFOperator of $(eltype(ifgf)) with range $(rowrange(ifgf)) × $(colrange(ifgf))")
    ctree = source_tree(ifgf)
    nodes  = collect(PreOrderDFS(ctree))
    leaves = collect(Leaves(ctree))
    println("\t number of nodes:                $(length(nodes))")
    println("\t number of leaves:               $(length(leaves))")
    println("\t number of active cones:         $(sum(x->length(active_cone_idxs(x)),nodes))")
    println("\t maximum number of active cones: $(maximum(x->length(active_cone_idxs(x)),nodes))")
end
Base.show(io::IO,::MIME"text/plain",ifgf::IFGFOp) = show(io,ifgf)
