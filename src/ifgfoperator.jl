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
    # do some planning for the fft
    @timeit_debug "planning fft" begin
        plan = FFTW.plan_r2r!(zeros(T,Ytree.data.p),FFTW.REDFT00;flags=FFTW.PATIENT)
    end
    # iterate over all nodes in source tree, visiting children before parents
    for node in AbstractTrees.PostOrderDFS(Ytree)
        length(node.loc_idxs) == 0 && continue
        # allocate necessary interpolation data and perform necessary
        # precomputations
        els = ElementIterator(node.data.cart_mesh)
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
            interps = _construct_chebyshev_interpolants(node,plan)
        end
        # build interp to far field map
        @timeit_debug "cone2far map" begin
            cone2farfield = _cone_to_far(node,Xtree)
        end
        @timeit_debug "cone2parent map" begin
            cone2parent   = _cone_to_parent(node)
        end
        # handle far targets by interpolation
        @timeit_debug "far interactions" begin
            # _compute_far_interaction!(C,A,B,node,interps)
            _compute_far_interaction!(C,A,B,node,interps,cone2farfield)
        end
        # transfer node's contribution to parent's interpolant
        isroot(node) && continue
        @timeit_debug "transfer to parent" begin
            # _transfer_to_parent!(C,A,B,node,interps)
            _transfer_to_parent!(C,A,B,node,interps,cone2parent)
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

function _construct_chebyshev_interpolants(node::SourceTree{N,Td,T},plan) where {N,Td,T}
    els     = ElementIterator(node.data.cart_mesh)
    interps = Dict{CartesianIndex{N},ChebPoly{N,T,Td}}()
    for I in node.data.active_idxs
        _,vals     = node.data.interp_data[I]
        rec        = els[I]
        poly       = chebfit!(vals,rec.low_corner,rec.high_corner,plan)
        push!(interps,I=>poly)
    end
    return interps
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
    for I in node.data.active_idxs
        nodes,vals = node.data.interp_data[I]
        for i in eachindex(nodes)
            xi = nodes[i] # cartesian coordinate of node
            # add sum of factored part of kernel to the interpolation node
            fact = 1/K(xi, xc)
            for j in J
                vals[i] += K(xi, Ypts[j])*fact*B[j]
            end
        end
    end
    return nothing
end

function _cone_to_far(node::SourceTree{N,Td,T},Xtree) where {N,Td,T}
    cone2farfield = Dict{CartesianIndex{N},Vector{Int}}()
    for far_target in node.data.far_list
        I = far_target.loc_idxs
        for i in I
            x       = Xtree.points[i]
            idxcone = cone_index(x, node)
            if haskey(cone2farfield,idxcone)
                push!(cone2farfield[idxcone],i)
            else
                cone2farfield[idxcone] = [i]
            end
        end
    end
    return cone2farfield
end

function _compute_far_interaction!(C,A,B,node,interps)
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
            poly    = interps[idxcone]
            # @assert s ∈ els[idxcone]
            C[i] += poly(s) * K(x, xc)
        end
    end
    return nothing
end

function _compute_far_interaction!(C,A,B,node,interps,cone2farfield)
    Xtree = target_tree(A)
    Xpts  = Xtree.points
    K     = kernel(A)
    bbox = node.bounding_box
    xc   = center(bbox)
    for (I,idxs) in cone2farfield
        poly    = interps[I]
        for i in idxs
            x       = Xpts[i]
            s       = cart2interp(x,node)
            # @assert s ∈ els[idxcone]
            C[i] += poly(s) * K(x, xc)
        end
    end
    return nothing
end

function _cone_to_parent(node::SourceTree{N,Td,T}) where {N,Td,T}
    cone2parent = Dict(I=>Dict{CartesianIndex{N},Vector{Int}}() for I in node.data.active_idxs)
    for idxinterp in node.parent.data.active_idxs
        nodes,_ = node.parent.data.interp_data[idxinterp]
        for idxval in eachindex(nodes)
            xi        = nodes[idxval]
            idxcone   = cone_index(xi, node)
            if haskey(cone2parent[idxcone],idxinterp)
                push!(cone2parent[idxcone][idxinterp],idxval)
            else
                cone2parent[idxcone][idxinterp] = [idxval]
            end
        end
    end
    return cone2parent
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
            scale = K(xi, xc) / K(xi, xc_parent)
            # TODO: this scaling factor can often be computed more efficiently
            # than two evaluations of the kernel followed by a division. We
            # should define a few methods for the kernel `K` such as
            # `scale_to_parent` and `analytic_factor` to make such evaluations faster.
            vals[idxval] += poly(si) * scale
        end
    end
    return nothing
end

function _transfer_to_parent!(C,A,B,node,interps,cone2parent)
    K     = kernel(A)
    Ytree = source_tree(A)
    T     = return_type(Ytree)
    bbox = node.bounding_box
    xc   = center(bbox)
    parent    = node.parent
    xc_parent = parent.bounding_box |> center
    for I in node.data.active_idxs
        poly = interps[I]
        for (Ip,idxs) in cone2parent[I]
            nodes,vals = parent.data.interp_data[Ip]
            for idxval in idxs
                xi        = nodes[idxval]
                si        = cart2interp(xi,node)
                # transfer block's interpolant to parent's interpolants
                scale = K(xi, xc) / K(xi, xc_parent)
                # TODO: this scaling factor can often be computed more efficiently
                # than two evaluations of the kernel followed by a division. We
                # should define a few methods for the kernel `K` such as
                # `scale_to_parent` and `analytic_factor` to make such evaluations faster.
                vals[idxval] += poly(si) * scale
            end
        end
    end
    return nothing
end
