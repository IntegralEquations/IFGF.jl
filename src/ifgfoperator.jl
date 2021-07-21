struct IFGFOperator{N,Td,T,K}
    kernel::K
    target_tree::TargetTree{N,Td}
    source_tree::SourceTree{N,Td,T}
end

kernel(op::IFGFOperator) = op.kernel
target_tree(op::IFGFOperator) = op.target_tree
source_tree(op::IFGFOperator) = op.source_tree

# function IFGFOperator(kernel,X,Y)
#     target_tree = initialize_target_tree(X,splitter)
#     target_tree = initialize_target_tree(X,splitter)
#     # create interaction lists and cones
#     initialize_interaction_list!(target_tree,source_tree,adm)
#     initialize_cone_list!(target_tree,source_tree,p,ds)
#     return IFGFOperator(targets,sources)
# end

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
        # handle near targets
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
        if isleaf(node)
            # leaf blocks are special in they are responsible for creating their
            # own cone interpolants as they cannot rely on their children to do
            # it
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
        end
        # handle far targets by interpolation
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
        # add node's contribution to parent's interpolants
        isroot(node) && continue
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
    end
    return C
end
