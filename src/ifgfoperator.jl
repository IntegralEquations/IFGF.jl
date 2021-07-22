struct IFGFOperator{N,Td,T,K}
    kernel::K
    target_tree::TargetTree{N,Td}
    source_tree::SourceTree{N,Td,T}
end

kernel(op::IFGFOperator) = op.kernel
target_tree(op::IFGFOperator) = op.target_tree
source_tree(op::IFGFOperator) = op.source_tree

function LinearAlgebra.mul!(C, A::IFGFOperator, B, a, b)
    K = kernel(A)
    # multiply at the beginning by `a` and `b` and be done with it
    rmul!(C, b)
    B   = a * B
    # extract the row/target and column/source trees for convenience
    Xtree = target_tree(A)
    Xpts  = Xtree.points
    Ytree = source_tree(A)
    Ypts  = Ytree.points
    # iterate over all nodes in source tree, visiting children before parents
    for node in AbstractTrees.PostOrderDFS(Ytree)
        bbox = node.bounding_box
        xc   = center(bbox)
        h    = radius(bbox)
        J    = node.loc_idxs
        @timeit_debug "near interactions" begin
            _compute_near_interaction!(C,A,B,node)
        end
        # leaf blocks need to create their own interpolants since they have no
        # children
        @timeit_debug "leaf interpolants" begin
            isleaf(node) && _compute_own_interpolant!(C,A,B,node)
        end
        # handle far targets by interpolation
        @timeit_debug "far interactions" begin
            _compute_far_interaction!(C,A,B,node)
        end
        # transfer node's contribution to parent's interpolant
        isroot(node) && continue
        @timeit_debug "transfer to parent" begin
            _transfer_to_parent!(C,A,B,node)
        end
    end
    return C
end

function _compute_near_interaction!(C,A::IFGFOperator,B,node)
    Xtree = target_tree(A)
    Xpts  = Xtree.points
    Ytree = source_tree(A)
    Ypts  = Ytree.points
    K     = kernel(A)
    J     = node.loc_idxs
    for near_target in node.data.near_list
        # near targets use direct evaluation
        I = near_target.loc_idxs
        for i in I
            iglob = Xtree.loc2glob[i]
            for j in J
                jglob = Ytree.loc2glob[j]
                C[iglob] += K(Xpts[iglob], Ypts[jglob]) * B[jglob]
            end
        end
    end
    return nothing
end

function _compute_own_interpolant!(C,A,B,node)
    Ytree = source_tree(A)
    Ypts  = Ytree.points
    K     = kernel(A)
    J     = node.loc_idxs
    bbox  = node.bounding_box
    xc    = center(bbox)
    for interp in values(node.data.interps)
        for idxval in CartesianIndices(interp.vals)
            si = interpolation_nodes(interp, idxval) # node in interpolation space
            xi = interp2cart(si, node) # cartesian coordinate of node
            # add sum of factored part of kernel to the interpolation node
            for j in J
                jglob = Ytree.loc2glob[j]
                interp.vals[idxval] += K(xi, Ypts[jglob]) / K(xi, xc) * B[jglob]
            end
        end
    end
    return nothing
end

function _compute_far_interaction!(C,A,B,node)
    Xtree = target_tree(A)
    Xpts  = Xtree.points
    Ytree = source_tree(A)
    K     = kernel(A)
    bbox = node.bounding_box
    xc   = center(bbox)
    for far_target in node.data.far_list
        I = far_target.loc_idxs
        for i in I
            iglob   = Xtree.loc2glob[i]
            x       = Xpts[iglob]
            idxcone = cone_index(x, node)
            s       = cart2interp(x,node)
            p       = node.data.interps[idxcone]
            C[iglob] += p(s) * K(x, xc)
        end
    end
    return nothing
end

function _transfer_to_parent!(C,A,B,node)
    K     = kernel(A)
    bbox = node.bounding_box
    xc   = center(bbox)
    parent    = node.parent
    xc_parent = parent.bounding_box |> center
    for (idxinterp,interp) in parent.data.interps
        for idxval in CartesianIndices(interp.vals)
            si_parent = interpolation_nodes(interp, idxval)  # node in interpolation space
            xi        = interp2cart(si_parent,parent)   # cartesian coordiante of node
            idxcone   = cone_index(xi, node)
            si        = cart2interp(xi,node)
            # if !haskey(node.data.interps,idxcone)
            #     return node,si_parent,xi,idxcone,idxval,si,idxinterp
            # end
            p         = node.data.interps[idxcone]
            # transfer block's interpolant to parent's interpolants
            interp.vals[idxval] += p(si) * K(xi, xc) / K(xi, xc_parent)
        end
    end
    return nothing
end
