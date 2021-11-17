
struct IFGFOperator{T} <: AbstractMatrix{T}
    kernel
    target_tree::TargetTree
    source_tree::SourceTree
end

rowrange(ifgf::IFGFOperator)         = Trees.index_range(ifgf.target_tree)
colrange(ifgf::IFGFOperator)         = Trees.index_range(ifgf.source_tree)

kernel(op::IFGFOperator) = op.kernel
target_tree(op::IFGFOperator) = op.target_tree
source_tree(op::IFGFOperator) = op.source_tree
function Base.size(op::IFGFOperator)
    sizeX = op |> target_tree |> root_elements |> length
    sizeY = op |> source_tree |> root_elements |> length
    return (sizeX,sizeY)
end

function Base.getindex(op::IFGFOperator,i::Int,j::Int)
    error()
    # m,n = size(op)
    # ei = zeros()
    # getindex(op*ei,j)
end

"""
    IFGFOperator(kernel,
                 Ypoints::Vector{V},
                 Xpoints::Vector{U};
                 splitter,
                 p_func,
                 ds_func) where {V,U}

Create the source and target tree-structures for clustering the source `Ypoints`
and target `Xpoints` using the splitting strategy of `splitter`. Both trees are
initialized (i.e. the interaction list and the cone list are computed).

# Arguments
- `kernel`: kernel of the IFGF operator.
- `Ypoints`: vector of source points.
- `Xpoints`: vector of observation points.
- `splitter`: splitting strategy.
- `p_func`: function `p_func(source_tree) = p` that returns the number of interpolation points
            per dimension in interpolation coordinates. In 3D, `p = (ns,nθ,nϕ)`.
            In 2D, `p = (ns,nθ)`.
- `ds_func`: function `ds_func(source_tree) = ds` that returns the size of the cone domains
             per dimension in interpolation coordinates. In 3D, `ds = (dss,dsθ,dsϕ)`.
             In 2D, `ds = (dss,dsθ)`.
"""
function IFGFOperator(kernel,
                      Ypoints::Vector{U},
                      Xpoints::Vector{V};
                      splitter,
                      p_func,
                      ds_func) where {U,V}
    # infer the type of the matrix
    T = hasmethod(return_type,Tuple{typeof(kernel)}) ? return_type(kernel) : Base.promote_op(kernel,U,V)
    assert_concrete_type(T)
    Tv = _density_type_from_kernel_type(T)
    source_tree = initialize_source_tree(;Ypoints,splitter,Xdatatype=U,Vdatatype=Tv)
    target_tree = initialize_target_tree(;Xpoints,splitter)
    compute_interaction_list!(source_tree,target_tree)
    compute_cone_list!(source_tree,p_func,ds_func)
    ifgf = IFGFOperator{T}(kernel, target_tree, source_tree)
    return ifgf
end

function _density_type_from_kernel_type(T)
    if T <: Number
        return T
    elseif T <: SMatrix
        m,n = size(T)
        return SVector{n,eltype(T)}
    else
        error("kernel type $T not recognized")
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::IFGFOperator, x::AbstractVector, a::Number, b::Number;global_index=true,threads=false,droptol=0)
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
    if threads
        _gemv_threaded!(y,K,rtree,ctree,x,T,droptol)
    else
        _gemv!(y,K,rtree,ctree,x,T,droptol)
    end
    # permute output
    global_index && invpermute!(y,loc2glob(rtree))

    return y
end

function _gemv_threaded!(y,K,target_tree,source_tree,x,T,droptol)
    Tv = if T <: Number
        T
    elseif  T<:SMatrix
        m,n = size(T)
        SVector{n,eltype(T)}
    end
    # do some planning for the fft
    @timeit_debug "planning fft" begin
        plan = FFTW.plan_r2r!(zeros(Tv,source_tree.data.p),FFTW.REDFT00;flags=FFTW.PATIENT)
    end
    # initialize a dictionary
    N  = ambient_dimension(target_tree)
    Td = Float64
    # the threading strategy is to partition the tree by depth, then go from
    # deepest to shallowest. Nodes of a fixed depth are mostly independent and
    # thus can spawn different tasks, (except for some "small" parts where they transfer
    # data to their parent or push into a dict). For that part, we use a lock.
    partition = Trees.partition_by_depth(source_tree)
    dmax = length(partition)
    for d in dmax:-1:1
        nodes = partition[d]
        n = length(nodes)
        for i in 1:n
            node = nodes[i]
            length(node) == 0 && continue
            # allocate necessary interpolation data and perform necessary
            # precomputations
            # @timeit_debug "allocate interpolant data" begin
                _allocate_data!(node)
                coneidxs2vals = node.data.coneidx2vals
            #    isempty(mempool) && # push more vals to mempool
            #    node2vals[node] = pop!(mempool)
            # end
            # @timeit_debug "near interactions" begin
                _compute_near_interaction!(y,K,target_tree,source_tree,x,node)
            # end
            # @timeit_debug "leaf interpolants" begin
                isleaf(node) && _compute_own_interpolant!(coneidxs2vals,K,source_tree,x,node)
            # end
            #
            # @timeit_debug "construct chebyshev interpolants" begin
                interps = _construct_chebyshev_interpolants(coneidxs2vals,node,plan,droptol)
            # end
            # @timeit_debug "far interactions" begin
                _compute_far_interaction!(y,K,target_tree,node,interps)
            # end
            # transfer node's contribution to parent's interpolant
            # @timeit_debug "transfer to parent" begin
                !isroot(node) && _transfer_to_parent!(K,node,interps)
            # end
            # free interpolation data
            empty!(coneidxs2vals)
        end
    end
end

function _gemv!(y,K,target_tree,source_tree,x,T,droptol)
    Tv = if T <: Number
        T
    elseif  T<:SMatrix
        m,n = size(T)
        SVector{n,eltype(T)}
    end
    # do some planning for the fft
    @timeit_debug "planning fft" begin
        plan = FFTW.plan_r2r!(zeros(Tv,source_tree.data.p),FFTW.REDFT00;flags=FFTW.PATIENT)
    end
    # initialize a dictionary
    N  = ambient_dimension(target_tree)
    Td = Float64
    node2dict = IdDict{typeof(source_tree), Dict{CartesianIndex{N},NodesAndVals{N,Td,Tv}}}()
    # iterate over all nodes in source tree, visiting children before parents
    for node in AbstractTrees.PostOrderDFS(source_tree)
        length(node) == 0 && continue
        # allocate necessary interpolation data and perform necessary
        # precomputations
        @timeit_debug "allocate interpolant data" begin
            _allocate_data!(node2dict, node)
            coneidxs2vals = node2dict[node]
        #    isempty(mempool) && # push more vals to mempool
        #    node2vals[node] = pop!(mempool)
        end
        @timeit_debug "near interactions" begin
            _compute_near_interaction!(y,K,target_tree,source_tree,x,node)
        end
        # leaf blocks need to update their own interpolant values since they
        # have no children
        @timeit_debug "leaf interpolants" begin
            isleaf(node) && _compute_own_interpolant!(coneidxs2vals,K,source_tree,x,node)
        end
        #
        @timeit_debug "construct chebyshev interpolants" begin
            interps = _construct_chebyshev_interpolants(coneidxs2vals,node,plan,droptol)
        end
        # build interp to far field map
        # @timeit_debug "cone2far map" begin
        #     cone2farfield = _cone_to_far(node,Xtree)
        # end
        # @timeit_debug "cone2parent map" begin
        #     cone2parent   = _cone_to_parent(node)
        # end
        # handle far targets by interpolation
        @timeit_debug "far interactions" begin
            _compute_far_interaction!(y,K,target_tree,node,interps)
        end
        # transfer node's contribution to parent's interpolant
        @timeit_debug "transfer to parent" begin
            !isroot(node) && _transfer_to_parent!(node2dict,K,node,interps)
        end
        # free interpolation data
        empty!(coneidxs2vals)
        delete!(node2dict,node)
        #vals = pop!(node2vals, node)
        #push!(mempool, vals)
    end
    return y
end

function _allocate_data!(node::SourceTree{N,Td,V,U,Tv}) where {N,Td,V,U,Tv}
    # leaves allocate their own interpolant vals
    if isleaf(node)
        coneidxs2vals = node.data.coneidx2vals
        for (I,rec) in active_cone_idxs_and_domains(node)
            cheb   = cheb2nodes_iter(node.data.p,rec)
            inodes = map(si->interp2cart(si,node),cheb)
            vals   = zeros(Tv,node.data.p)
            push!(coneidxs2vals, I => (inodes,vals))
        end
    end
    # allocate parent data (if needed)
    if !isroot(node)
        parent_node = parent(node)
        coneidxs2vals_parent = parent_node.data.coneidx2vals
        if isempty(coneidxs2vals_parent)
            for (I,rec) in active_cone_idxs_and_domains(parent_node)
                cheb   = cheb2nodes_iter(node.parent.data.p,rec)
                inodes = map(si->interp2cart(si,node.parent),cheb)
                vals   = zeros(Tv,node.data.p)
                push!(coneidxs2vals_parent, I => (inodes,vals))
            end
        end
    end
    return nothing
end

function _compute_near_interaction!(C,K,target_tree,source_tree,B,node)
    Xpts = target_tree |> root_elements
    Ypts = source_tree |> root_elements
    for near_target in near_list(node)
        # near targets use direct evaluation
        for i in index_range(near_target)
            for j in index_range(node)
                C[i] += K(Xpts[i], Ypts[j]) * B[j]
            end
        end
    end
    return nothing
end

function _compute_own_interpolant!(coneidxs2vals::Dict,K,source_tree,B,node)
    Ypts  = source_tree |> root_elements
    for I in active_cone_idxs(node)
        nodes,vals = coneidxs2vals[I]
        for i in eachindex(nodes)
            xi = nodes[i] # cartesian coordinate of node
            # add sum of factored part of kernel to the interpolation node
            fact = 1/centered_factor(K,xi,node) # defaults to 1/K
            for j in index_range(node)
                y = Ypts[j]
                vals[i] += K(xi, y)*fact*B[j]
            end
        end
    end
    return nothing
end

function _construct_chebyshev_interpolants(coneidxs2vals::Dict{CartesianIndex{N},NodesAndVals{N,Td,T}},
                                           node::SourceTree{N,Td},plan,tol) where {N,Td,T}
    # interps = Dict{CartesianIndex{N},TensorLagInterp{N,Td,T}}()
    interps = Dict{CartesianIndex{N},ChebPoly{N,T,Td}}()
    p       = node.data.p
    for (I,rec) in active_cone_idxs_and_domains(node)
        _,vals    = coneidxs2vals[I]
        lc        = low_corner(rec)
        hc        = high_corner(rec)
        # nodes1d   = ntuple(d->cheb2nodes(p[d],lc[d],hc[d]),N)
        # weights1d = ntuple(d->cheb2weights(p[d]),N)
        # poly      = TensorLagInterp(vals,nodes1d,weights1d)
        poly      = chebinterp!(vals,lc,hc,plan;tol)
        push!(interps,I=>poly)
    end
    return interps
end

function _compute_far_interaction!(C,K,target_tree,node,interps)
    Xpts  = target_tree |> root_elements
    # els = cone_domains(node)
    for far_target in far_list(node)
        for i in index_range(far_target)
            x       = Xpts[i]
            idxcone = cone_index(coords(x),node)
            s       = cart2interp(coords(x),node)
            poly    = interps[idxcone]
            # @assert s ∈ els[idxcone]
            C[i] += poly(s) * centered_factor(K,x,node)
        end
    end
    return nothing
end


function _transfer_to_parent!(K,node,interps)
    node_parent = parent(node)
    coneidxs2vals_parent = node_parent.data.coneidx2vals
    for idxinterp in active_cone_idxs(node_parent)
        nodes,vals = coneidxs2vals_parent[idxinterp]
        for idxval in eachindex(nodes)
            xi = nodes[idxval]
            idxcone   = cone_index(xi, node)
            si        = cart2interp(xi,node)
            poly      = interps[idxcone]
            # transfer block's interpolant to parent's interpolants
            vals[idxval] += poly(si) * transfer_factor(K,xi,node)
        end
    end
    return nothing
end

# generic fallback. May need to be overloaded for specific kernels.
function centered_factor(K,x,ysource::SourceTree)
    yc = center(ysource)
    return K(x,yc)
end

# generic fallback. May need to be overloaded for specific kernels.
function transfer_factor(K,x,ysource::SourceTree)
    yparent = parent(ysource)
    return centered_factor(K,x,ysource)/centered_factor(K,x,yparent)
end

function Base.show(io::IO,ifgf::IFGFOperator)
    println(io,"IFGFOperator of $(eltype(ifgf)) with range $(rowrange(ifgf)) × $(colrange(ifgf))")
    ctree = source_tree(ifgf)
    nodes  = collect(PreOrderDFS(ctree))
    leaves = collect(Leaves(ctree))
    println("\t number of nodes:                $(length(nodes))")
    println("\t number of leaves:               $(length(leaves))")
    println("\t number of active cones:         $(sum(x->length(x.data.active_cone_idxs),nodes))")
    println("\t maximum number of active cones: $(maximum(x->length(x.data.active_cone_idxs),nodes))")
end
Base.show(io::IO,::MIME"text/plain",ifgf::IFGFOperator) = show(io,ifgf)
