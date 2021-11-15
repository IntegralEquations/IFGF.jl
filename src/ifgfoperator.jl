
"""
    NodesAndVals{N,Td,T} = Tuple{Array{SVector{N,Td},N},Array{T,N}}

For storing the Chebyshev nodes and interpolation values.
"""
const NodesAndVals{N,Td,T} = Tuple{Array{SVector{N,Td},N},Array{T,N}}

struct IFGFOperator{N,Td,K,U,V}
    kernel::K
    target_tree::TargetTree{N,Td,U}
    source_tree::SourceTree{N,Td,V,U}
end

kernel(op::IFGFOperator) = op.kernel
target_tree(op::IFGFOperator) = op.target_tree
source_tree(op::IFGFOperator) = op.source_tree
Base.size(op::IFGFOperator,i) = size(op)[i]
function Base.size(op::IFGFOperator)
    sizeX = op |> target_tree |> root_elements |> length
    sizeY = op |> source_tree |> root_elements |> length
    return (sizeX,sizeY)
end
function Base.:*(A::IFGFOperator{N,Td}, B::AbstractArray{T,NB}) where {N,Td,T,NB}
    # FIX: Ctype is not equal to T
    Ctype = T
    if NB == 1
        C = Vector{Ctype}(undef, size(A,1))
    elseif NB == 2
        C = Matrix{Ctype}(undef, size(A,1), size(B,2))
    else
        notimplemented()
    end
    return mul!(C, A, B, true, false)
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
                      Ypoints::Vector{V},
                      Xpoints::Vector{U};
                      splitter,
                      p_func,
                      ds_func,
                      _profile=false) where {V,U}
    source_tree = initialize_source_tree(;Ypoints,splitter,Xdatatype=U)
    target_tree = initialize_target_tree(;Xpoints,splitter)
    compute_interaction_list!(source_tree,target_tree)
    if !_profile
        compute_cone_list!(source_tree,p_func,ds_func)
    else
        @hprofile compute_cone_list!(source_tree,p_func,ds_func)
    end
    ifgf = IFGFOperator(kernel, target_tree, source_tree)
    return ifgf
end

function LinearAlgebra.mul!(C, A::IFGFOperator{N,Td,K,U,V}, B::AbstractVector{T}, a, b) where {N,Td,K,U,V,T}
    # check compatible sizes
    @assert size(A) == (length(C),length(B))
    # extract the row/target and column/source trees for convenience
    Xtree = target_tree(A)
    Ytree = source_tree(A)
    # since the IFGFOperator represents A = Pr*H*Pc, where Pr and Pc are row and
    # column permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end.
    # permute input
    B = B[loc2glob(Ytree)]  # this makes a copy of B
    C = permute!(C,loc2glob(Xtree))
    # multiply at the beginning by `a` and `b` and be done with it
    rmul!(C,b)
    rmul!(B,a)
    # iterate over all nodes in source tree, visiting children before parents
    node2dict = IdDict{SourceTree{N,Td,V,U}, Dict{CartesianIndex{N},NodesAndVals{N,Td,T}}}()
    for node in AbstractTrees.PostOrderDFS(Ytree)
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
            _compute_near_interaction!(C,A,B,node)
        end
        # leaf blocks need to update their own interpolant values since they
        # have no children
        @timeit_debug "leaf interpolants" begin
            isleaf(node) && _compute_own_interpolant!(coneidxs2vals,A,B,node)
        end
        #
        @timeit_debug "construct chebyshev interpolants" begin
            interps = _construct_chebyshev_interpolants(coneidxs2vals,node)
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
            _compute_far_interaction!(C,A,node,interps)
        end
        # transfer node's contribution to parent's interpolant
        @timeit_debug "transfer to parent" begin
            !isroot(node) && _transfer_to_parent!(node2dict,A,node,interps)
        end
        # free interpolation data
        empty!(coneidxs2vals)
        delete!(node2dict,node)
        #vals = pop!(node2vals, node)
        #push!(mempool, vals)
    end
    # permute output
    invpermute!(C,loc2glob(Xtree))
    return C
end
function LinearAlgebra.mul!(C, A::IFGFOperator, B::AbstractMatrix, a, b)
    # TODO: multithreading
    # check compatible sizes
    @assert size(C) == (size(A,1),size(B,2))
    @assert size(A,2) == size(B,1)
    # compute product for each column
    for (Bcol,Ccol) in zip(eachcol(B),eachcol(C))
        mul!(Ccol, A, Bcol, a, b)
    end
    return C
end

function _allocate_data!(node2dict::IdDict{<:SourceTree, Dict{CartesianIndex{N},NodesAndVals{N,Td,T}}},
                         node::SourceTree) where {N,Td,T}
    # leaves allocate their own interpolant vals
    if isleaf(node)
        coneidxs2vals = Dict{CartesianIndex{N},NodesAndVals{N,Td,T}}()
        push!(node2dict, node => coneidxs2vals)
        for (I,rec) in active_cone_idxs_and_domains(node)
            cheb   = cheb2nodes_iter(node.data.p,rec)
            inodes = map(si->interp2cart(si,node),cheb)
            vals   = zeros(T,node.data.p)
            push!(coneidxs2vals, I => (inodes,vals))
        end
    end
    # allocate parent data (if needed)
    if !isroot(node) && !haskey(node2dict, parent(node))
        parent_node = parent(node)
        coneidxs2vals_parent = Dict{CartesianIndex{N},NodesAndVals{N,Td,T}}()
        push!(node2dict, parent_node => coneidxs2vals_parent)
        for (I,rec) in active_cone_idxs_and_domains(parent_node)
            cheb   = cheb2nodes_iter(node.parent.data.p,rec)
            inodes = map(si->interp2cart(si,node.parent),cheb)
            vals   = zeros(T,node.data.p)
            push!(coneidxs2vals_parent, I => (inodes,vals))
        end
    end
    return nothing
end

function _compute_near_interaction!(C,A::IFGFOperator,B,node)
    Xpts = A |> target_tree |> root_elements
    Ypts = A |> source_tree |> root_elements
    K    = kernel(A)
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

function _compute_own_interpolant!(coneidxs2vals::Dict,A::IFGFOperator,B,node)
    Ypts  = A |> source_tree |> root_elements
    K     = kernel(A)
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
                                           node::SourceTree{N,Td}) where {N,Td,T}
    interps = Dict{CartesianIndex{N},TensorLagInterp{N,Td,T}}()
    p       = node.data.p
    for (I,rec) in active_cone_idxs_and_domains(node)
        _,vals    = coneidxs2vals[I]
        lc        = low_corner(rec)
        uc        = high_corner(rec)
        nodes1d   = ntuple(d->cheb2nodes(p[d],lc[d],uc[d]),N)
        weights1d = ntuple(d->cheb2weights(p[d]),N)
        poly      = TensorLagInterp(vals,nodes1d,weights1d)
        push!(interps,I=>poly)
    end
    return interps
end

function _compute_far_interaction!(C,A::IFGFOperator,node,interps)
    Xpts  = A |> target_tree |> root_elements
    K     = kernel(A)
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

function _transfer_to_parent!(node2dict,A::IFGFOperator,node,interps)
    K = kernel(A)
    node_parent = parent(node)
    coneidxs2vals_parent = node2dict[node_parent]
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
