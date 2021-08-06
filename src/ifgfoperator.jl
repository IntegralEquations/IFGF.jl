struct IFGFOperator{N,Td,T,K}
    kernel::K
    target_tree::TargetTree{N,Td}
    source_tree::SourceTree{N,Td,T}
end

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
        @timeit_debug "prepare interpolants" begin
            # precompute nodes and weights to make interpolation faster
            for interp in values(node.data.interps)
                precompute_nodes_and_weights!(interp)
            end
            # leaves allocate their own interpolante vals
            if isleaf(node)
                for interp in values(node.data.interps)
                    p = length.(interp.nodes1d)
                    interp.vals = zeros(T,p)
                end
            end
            # allocate parent data (if needed)
            isroot(node) && continue
            for interp in values(node.parent.data.interps)
                isempty(interp.vals) || continue
                p = length.(interp.nodes1d)
                interp.vals = zeros(T,p)
            end
        end
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
        # free interpolation data
        for interp in values(node.data.interps)
            interp.vals     = zeros(T,ntuple(i->0,N))
            interp._nodes   = zeros(SVector{N,Td},ntuple(i->0,N))
            interp._weights = zeros(Td,ntuple(i->0,N))
        end
    end
    # permute output
    permute!(C,Xtree.glob2loc)
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
            for j in J
                C[i] += K(Xpts[i], Ypts[j]) * B[j]
            end
        end
    end
    return nothing
end

function _compute_own_interpolant!(C,A,B,node)
    Ytree = source_tree(A)
    T     = return_type(Ytree)
    Ypts  = Ytree.points
    K     = kernel(A)
    J     = node.loc_idxs
    bbox  = node.bounding_box
    xc    = center(bbox)
    for interp in values(node.data.interps)
        for idxval in CartesianIndices(interp)
            si = interpolation_nodes(interp,idxval) # node in interpolation space
            xi = interp2cart(si, node) # cartesian coordinate of node
            # add sum of factored part of kernel to the interpolation node
            fact = 1/K(xi, xc)
            for j in J
                interp.vals[idxval] += K(xi, Ypts[j])*fact*B[j]
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
    els = ElementIterator(node.data.cart_mesh)
    for far_target in node.data.far_list
        I = far_target.loc_idxs
        for i in I
            x       = Xpts[i]
            idxcone = cone_index(x, node)
            s       = cart2interp(x,node)
            p       = node.data.interps[idxcone]
            # @assert s ∈ els[idxcone]
            C[i] += p(s,Val(:fast)) * K(x, xc)
        end
    end
    return nothing
end

function _transfer_to_parent!(C,A,B,node)
    K     = kernel(A)
    Ytree = source_tree(A)
    T     = return_type(Ytree)
    bbox = node.bounding_box
    xc   = center(bbox)
    parent    = node.parent
    xc_parent = parent.bounding_box |> center
    els = ElementIterator(node.data.cart_mesh)
    for (idxinterp,interp) in parent.data.interps
        for idxval in CartesianIndices(interp.vals)
            si_parent = interpolation_nodes(interp, idxval)  # node in interpolation space
            xi        = interp2cart(si_parent,parent)   # cartesian coordiante of node
            idxcone   = cone_index(xi, node)
            si        = cart2interp(xi,node)
            p         = node.data.interps[idxcone]
            # transfer block's interpolant to parent's interpolants
            # @assert si ∈ els[idxcone]
            scale = K(xi, xc) / K(xi, xc_parent)
            # d1 = norm(xi-xc)
            # d2 = norm(xi-xc_parent)
            # scale = ComplexF64(exp(im*4π*(d1-d2))*d2/d1)
            interp.vals[idxval] += p(si,Val(:fast)) * scale
        end
    end
    return nothing
end

function test_source_tree(tree)
    for node in AbstractTrees.PostOrderDFS(tree)
        for interp in values(node.data.interps)
            vmin,vmax = extrema(abs,interp.vals)
            if vmin == vmax == 0
                @info vmin,vmax
            end
        end
    end
end
