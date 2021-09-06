struct IFGFOperator{N,Td,T,K,U,V}
    kernel::K
    target_tree::TargetTree{N,Td}
    source_tree::SourceTree{N,Td,T}
    target_dofs::Vector{U}
    source_dofs::Vector{V}
end
IFGFOperator(K,target::TargetTree,source::SourceTree) = IFGFOperator(K,target,source,target.points,source.points)

kernel(op::IFGFOperator) = op.kernel
target_tree(op::IFGFOperator) = op.target_tree
source_tree(op::IFGFOperator) = op.source_tree

function LinearAlgebra.mul!(C, A::IFGFOperator{N,Td,T}, B, a, b) where {N,Td,T}
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
    B         = B[Ytree.loc2glob]
    C         = permute!(C,Xtree.loc2glob)
    # multiply at the beginning by `a` and `b` and be done with it
    rmul!(C,b)
    rmul!(B,a)
    # iterate over all nodes in source tree, visiting children before parents
    for node in AbstractTrees.PostOrderDFS(Ytree)
        length(node.loc_idxs) == 0 && continue
        # allocate necessary interpolation data and perform necessary
        # precomputations
        @timeit_debug "allocate interpolant data" begin
            _allocate_data!(node)
        end
        @timeit_debug "near interactions" begin
            _compute_near_interaction!(C,A,B,node)
        end
        # leaf blocks need to update their own interpolant values since they
        # have no children
        @timeit_debug "leaf interpolants" begin
            isleaf(node) && _compute_own_interpolant!(C,A,B,node)
        end
        #
        @timeit_debug "construct chebyshev interpolants" begin
            interps = _construct_chebyshev_interpolants(node)
        end
        # handle far targets by interpolation
        @timeit_debug "far interactions" begin
            _compute_far_interaction!(C,A,B,node,interps)
        end
        # transfer node's contribution to parent's interpolant
        isroot(node) && continue
        @timeit_debug "transfer to parent" begin
            _transfer_to_parent!(C,A,B,node,interps)
        end
        # free interpolation data
        empty!(node.data.interp_data)
    end
    # permute output
    permute!(C,Xtree.glob2loc)
    return C
end

function _allocate_data!(node::SourceTree{N,Td,T}) where {N,Td,T}
    # leaves allocate their own interpolant vals
    els = ElementIterator(node.data.cart_mesh)
    if isleaf(node)
        for I in node.data.active_idxs
            rec = els[I]
            cheb   = cheb2nodes_iter(node.data.p,rec.low_corner,rec.high_corner)
            inodes = map(si->interp2cart(si,node),cheb)
            vals   = zeros(T,node.data.p)
            node.data.interp_data[I] = (inodes, vals)
        end
    end
    # allocate parent data (if needed)
    if !isroot(node)
        els_parent = ElementIterator(node.parent.data.cart_mesh)
        for I in node.parent.data.active_idxs
            haskey(node.parent.data.interp_data,I) && continue # already initialized
            rec = els_parent[I]
            cheb   = cheb2nodes_iter(node.parent.data.p,rec.low_corner,rec.high_corner)
            inodes = map(si->interp2cart(si,node.parent),cheb)
            vals   = zeros(T,node.data.p)
            node.parent.data.interp_data[I] = (inodes,vals)
        end
    end
    return node
end

function _construct_chebyshev_interpolants(node::SourceTree{N,Td,T}) where {N,Td,T}
    els     = ElementIterator(node.data.cart_mesh)
    interps = Dict{CartesianIndex{N},TensorLagInterp{N,Td,T}}()
    p       = node.data.p
    for I in node.data.active_idxs
        _,vals     = node.data.interp_data[I]
        rec        = els[I]
        lc = rec.low_corner
        uc = rec.high_corner
        nodes1d   = ntuple(d->cheb2nodes(p[d],lc[d],uc[d]),N)
        weights1d = ntuple(d->cheb2weights(p[d]),N)
        poly      = TensorLagInterp(vals,nodes1d,weights1d)
        push!(interps,I=>poly)
    end
    return interps
end

function _compute_near_interaction!(C,A::IFGFOperator,B,node)
    Xpts  = A.target_dofs
    Ypts  = A.source_dofs
    K     = kernel(A)
    J     = node.loc_idxs
    for near_target in node.data.near_list
        # near targets use direct evaluation
        I = near_target.loc_idxs
        for i in I
            for j in J
                C[i] += K(Xpts[i], Ypts[j]) * B[j]
            end
        end
    end
    return nothing
end

function _compute_own_interpolant!(C,A::IFGFOperator,B,node)
    Ypts  = A.source_dofs
    K     = kernel(A)
    J     = node.loc_idxs
    bbox  = node.bounding_box
    xc    = center(bbox)
    for I in node.data.active_idxs
        nodes,vals = node.data.interp_data[I]
        for i in eachindex(nodes)
            xi = nodes[i] # cartesian coordinate of node
            # add sum of factored part of kernel to the interpolation node
            fact = 1/centered_factor(K,xi,xc) # defaults to 1/K
            for j in J
                y = Ypts[j]
                vals[i] += K(xi, y)*fact*B[j]
            end
        end
    end
    return nothing
end

function _compute_far_interaction!(C,A,B,node,interps)
    Xpts  = A.target_dofs
    K     = kernel(A)
    bbox = node.bounding_box
    xc   = center(bbox)
    els = ElementIterator(node.data.cart_mesh)
    for far_target in node.data.far_list
        I = far_target.loc_idxs
        for i in I
            x       = Xpts[i]
            idxcone = cone_index(coords(x), node)
            s       = cart2interp(coords(x),node)
            poly    = interps[idxcone]
            # @assert s ∈ els[idxcone]
            C[i] += poly(s) * centered_factor(K,x,xc)
        end
    end
    return nothing
end

function _transfer_to_parent!(C,A,B,node,interps)
    K     = kernel(A)
    Ytree = source_tree(A)
    T     = return_type(Ytree)
    bbox = node.bounding_box
    xc   = center(bbox)
    parent    = node.parent
    xc_parent = parent.bounding_box |> center
    for idxinterp in parent.data.active_idxs
        nodes,vals = parent.data.interp_data[idxinterp]
        for idxval in eachindex(nodes)
            xi = nodes[idxval]
            idxcone   = cone_index(xi, node)
            si        = cart2interp(xi,node)
            poly      = interps[idxcone]
            # transfer block's interpolant to parent's interpolants
            vals[idxval] += poly(si) * transfer_factor(K,xi,xc,xc_parent)
        end
    end
    return nothing
end

# generic fallback. May need to be overloaded for specific kernels.
function centered_factor(K,x,yc)
    return K(x,yc)
end

# generic fallback. May need to be overloaded for specific kernels.
function transfer_factor(K,x,yc,yc_parent)
    return centered_factor(K,x,yc)/centered_factor(K,x,yc_parent)
end
