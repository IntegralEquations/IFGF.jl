const ETA = 0.9

mutable struct SourceTreeData{N,Td,T}
    cart_mesh::UniformCartesianMesh{N,Td} # mesh of [0,η] × [0,π] × (-π,π], with η < 1
    interps::Dict{CartesianIndex{N},ChebInterp{N,Td,T}}
    far_list::Vector{TargetTree{N,Td}}
    near_list::Vector{TargetTree{N,Td}}
end
function SourceTreeData{N,Td,T}() where {N,Td,T}
    if N == 2
        domain = HyperRectangle{2,Td}((0,0),(ETA,2π))
        msh    = UniformCartesianMesh(domain,(1,1))
    elseif N == 3
        domain = HyperRectangle{3,Td}((0,0,0),(ETA,π,2π))
        msh = UniformCartesianMesh(domain,(1,1,1))
    else
        notimplemented()
    end
    interps   = Dict{CartesianIndex{N},ChebInterp{N,Td,T}}()
    far_list  = Vector{TargetTree{N,Td}}()
    near_list = Vector{ClusterTree{N,Td,T}}()
    SourceTreeData{N,Td,T}(msh,interps,far_list,near_list)
end

const SourceTree{N,Td,T} = ClusterTree{N,Td,SourceTreeData{N,Td,T}}

return_type(::SourceTree{N,Td,T}) where {N,Td,T} = T

"""
    interp2cart(si,source::SourceTree)

Map interpolation coordinate `s` into cartesian coordinates.
"""
function interp2cart(si,source::SourceTree{N}) where {N}
    bbox = source.bounding_box
    xc   = center(bbox)
    h    = radius(bbox)
    if N == 2
        s,θ  = si
        @assert 0 ≤ θ < 2π
        r    = h/s
        x,y  = pol2cart(r,θ-π) #
        return SVector(x,y) + xc
    elseif N == 3
        s,θ,ϕ  = si
        @assert 0 ≤ θ < π
        @assert 0 ≤ ϕ < 2π
        r      = h/s
        x,y,z    = sph2cart(r,θ,ϕ-π)
        return SVector(x,y,z) + xc
    else
        notimplemented()
    end
end

"""
    cart2interp(x,source::SourceTree)

Map cartesian coordinates `x` into (local) interpolation coordinates s.
"""
function cart2interp(x,source::SourceTree{N}) where {N}
    bbox = source.bounding_box
    xp   = x - center(bbox)
    h    = radius(bbox)
    if N == 2
        r,θ  = cart2pol(xp...)
        s = h/r
        return SVector(s,θ+π)
    elseif N == 3
        r,θ,ϕ  = cart2sph(xp...)
        s = h/r
        return SVector(s,θ,ϕ+π)
    else
        notimplemented()
    end
end

"""
    interpolation_domain(tree::SourceTree)

HyperRectangle where interpolation in the `(s,θ,φ)` variables (or (`s,θ`) in two
dimensions) take place.
"""
function interpolation_domain(::SourceTree{N,Td}) where {N,Td}
    if N == 2
        domain = HyperRectangle{2,Td}((0,0),(ETA,2π))
    elseif N == 3
        domain = HyperRectangle{3,Td}((0,0,0),(ETA,π,2π))
    else
        notimplemented()
    end
    return domain
end

cart_mesh(t::SourceTree) = t |> data |> cart_mesh
interps(t::SourceTree)   = t |> data |> interps
far_list(t::SourceTree)   = t |> data |> far_list
near_list(t::SourceTree)   = t |> data |> near_list

"""
    initialize_source_tree(;points,splitter)

Create the tree-structure for clustering the source `points` using the splitting
strategy of `splitter`, returning a `SourceTree` with an empty `data` field
(i.e. `data = SourceTreeData()`).
"""
function initialize_source_tree(;points::Vector{SVector{N,Td}},splitter,datatype) where {N,Td}
    tree = SourceTree{N,Td,datatype}(points,splitter)
    return tree
end

"""
    compute_interaction_list!(target_tree,source_tree,adm)

For each node in `source_tree`, compute its `near_list` and `far_list`. The
`near_list` contains all nodes in `target_tree` for which the interaction must be
computed using the direct summation.  The `far_list` includes all nodes in
`target_tree` for which an accurate interpolant in available in the source node.

The `adm` function is called through `adm(target,source)` to determine if
`target` is in the `far_list` of source. When that is not the case, we recurse
on the children of both `target` and `source` (unless either is a leaf, in which
case `target` is added to the `near_list` of source).
"""
function compute_interaction_list!(target::TargetTree,source::SourceTree{N,Td,T},adm) where {N,Td,T}
    if adm(target,source)
        push!(source.data.far_list,target)
    elseif isleaf(target) && isleaf(source)
        push!(source.data.near_list,target)
    elseif isleaf(target)
        # recurse on children of source
        for source_child in children(source)
            compute_interaction_list!(target,source_child,adm)
        end
    elseif isleaf(source)
        # recurse on children of target
        for target_child in children(target)
            compute_interaction_list!(target_child,source,adm)
        end
    else
        # recurse on children of target and source
        for source_child in children(source)
            for target_child in children(target)
                compute_interaction_list!(target_child,source_child,adm)
            end
        end
    end
    return nothing
end

function admissible(target,source,η=ETA)
    # compute distance between source center and target box
    xc   = center(source.bounding_box)
    h    = radius(source.bounding_box)
    bbox = target.bounding_box
    dc   = distance(xc,bbox)
    # if target box is outside a sphere of radius h/η, consider it admissible.
    # The same parameter η is used to construct the interpolation domain where
    # `s=h/r ∈ (0,η)` with η<1. This assures that target points which must be
    # interpolated are inside the interpolation domain.
    dc > h/η
end

"""
    compute_cone_list!(target,source,order,p,ds)

For each node in `source`, compute the cones which are necessary to cover
both the target points on its `far_list` as well as the interpolation points on
its parent's cones. The arguments `p` and `ds` are used to compute
an appropriate interpolation order and meshsize in interpolation space.

The algorithm starts at the root of the `source` tree, and travels down to the
leaves.
"""
function compute_cone_list!(tree::SourceTree,p,ds_func)
    for source in AbstractTrees.PreOrderDFS(tree)
        T = return_type(source)
        ds             = ds_func(source)
        data           = source.data
        domain         = interpolation_domain(source)
        data.cart_mesh = UniformCartesianMesh(domain;step=ds)
        els            = ElementIterator(data.cart_mesh)
        dict           = data.interps
        # initialize all cones needed to cover far field
        for far_target in data.far_list
            I = far_target.loc_idxs
            for i in I
                iglob = far_target.loc2glob[i]
                x     = far_target.points[iglob] # target point
                I     = cone_index(x,source)
                haskey(dict,I) && continue
                vals   = zeros(T,p)
                interp = ChebInterp(vals,els[I])
                push!(dict,I=>interp)
            end
        end
        # if not a root, also create all cones needed to cover interpolation points of parents
        if !isroot(source)
            parent = source.parent
            for (idxinterp,interp) in parent.data.interps
                for idxval in CartesianIndices(interp.vals)
                    # interpolation node in interpolation space
                    si = interpolation_nodes(interp,idxval)
                    # interpolation node in physical space
                    xi = interp2cart(si,parent)
                    # mesh index for interpolation point
                    I = cone_index(xi,source)
                    # initialize cone if needed
                    haskey(dict,I) && continue
                    vals   = zeros(T,p)
                    inew = ChebInterp(vals,els[I])
                    push!(dict,I=>inew)
                end
            end
        end
    end
    # now that all necessary cones are created, move to children
    # for child in source.children
    #     compute_cone_list!(child,p,ds_func)
    # end
    return tree
end

function cone_index(x,tree::SourceTree)
    msh = tree.data.cart_mesh
    s   = cart2interp(x,tree)
    Δs  = step(msh)
    N   = length(s)
    I  = ntuple(N) do n
        q = s[n]/Δs[n]
        ceil(Int,q)
    end
    CartesianIndex(I)
end

function clear_interpolants!(source::SourceTree)
    for interp in values(source.data.interps)
        fill!(interp.vals,0)
    end
    if !isleaf(source)
        for child in children(source)
            clear_interpolants!(child)
        end
    end
    return source
end

"""
    interpolation_points(s::SourceTree,I::CartesianIndex)

Return the interpolation points, in cartesian coordinates, for the cone
interpolant of index `I`.
"""
function interpolation_points(s::SourceTree,I::CartesianIndex)
    data   = s.data
    els    = ElementIterator(data.cart_mesh)
    rec    = els[I]
    xc     = center(rec)
    R      = radius(rec)
    interp = data.interps[I]
    iter   = CartesianIndices(interp)
    map(iter) do I
        si = interpolation_nodes(interp,I)
        xi = interp2cart(si,s)
        return xi
    end
end

# kernel(s::SourceTree) = s.kernel
# bounding_box(s::SourceTree) = s.bounding_box

# function evaluate_near_field(s::SourceTree,x)
#     K = kernel(s)
#     I = s.loc_idxs
#     sum(I) do i
#         K(x,s.points[i])*s.density[i]
#     end
# end

# function evaluate_far_field(s::SourceTree,x)
#     K = kernel(s)
#     yc = center(s)
#     I = conical_index(x,s)
#     p = getcone(s,I)
#     K(x,ys)*p(x)
# end

# function interpolation_coordinate(x::SVector{3},tree::SourceTree)
#     xp = x - center(bounding_box(tree))
#     R  = radius(bounding_box(tree))
#     r,θ,ϕ  = cart2sph(xp...)
#     s = R/r
#     return SVector(s,θ,ϕ)
# end

# function interpolation_coordinate(x::SVector{2},tree::SourceTree)
#     xp  = x - center(tree)
#     R = radius(bounding_box(tree))
#     r,θ = cart2pol(xp...)
#     s = R/r
#     return s,θ
# end

# # function compute_interaction_list!(tree::IFGFSourceTree{N,Td,T},targets::ClusterTree{N,Td},adm::Function) where {N,Td,T}
# #     Y = tree.source_tree
# #     X = targets
# #     if adm(X,Y)
# #         push!(tree.far_list,X)
# #     else
# #         if HMatrices.isleaf(X) || HMatrices.isleaf(Y)
# #             push!(tree.near_list,X)
# #         else
# #             for Yc in Y.children
# #                 child = IFGFSourceTree{N,Td,T}(;source_tree=Yc,parent=[tree])
# #                 push!(tree.children,child)
# #                 for Xc in X.children
# #                     compute_interaction_list!(child,Xc,adm)
# #                 end
# #             end
# #         end
# #     end
# #     return tree
# # end

# # function compute_cone_list!(tree::IFGFSourceTree{N,Td,T}) where {N,Td,T}
# #     S = Set{NTuple{N,Int}}() # index set of all active cones
# #     # tag all cones needed to cover far field interaction
# #     for A in tree.far_list
# #         for i in A.loc_idxs
# #             iglob = A.loc2glob[i]
# #             x     = A.points[iglob] # target point
# #             I     = compute_canonical_index(x,tree)
# #             push!(S,I)
# #         end
# #     end
# #     # if not a root, tag all cones needed to cover interpolation points of parents
# #     if !isroot(tree)
# #         Bp = tree.parent
# #         for C in Bp.cones
# #             for x in interpolation_points(C)
# #                 I   = compute_canonical_index(x,tree)
# #                 push!(S,I)
# #             end
# #         end
# #     end
# #     # now that all active cones are known, create them allocating the necessary memory
# #     for I in S
# #         lb = I .* Δs
# #         ub = lb + Δs
# #         push!(tree.cones,I=>ConeInterpolant{N,Td,T}(lb,ub,P))
# #     end
# #     # recurse on children
# #     for Bc in tree.children
# #         compute_cone_list!(Bc)
# #     end
# #     return tree
# # end

# # isleaf(t::IFGFSourceTree) = isempty(t.children)
# # isroot(t::IFGFSourceTree) = isempty(t.parent)

struct PlotInterpolationPoints end

@recipe function f(::PlotInterpolationPoints,tree::SourceTree)
    for (I,interp) in tree.data.interps
        @series begin
            seriestype := :scatter
            markercolor --> :black
            markershape --> :cross
            WavePropBase.IO.PlotPoints(), interpolation_points(tree,I)
        end
    end
end

@recipe function f(tree::SourceTree)
    legend --> false
    grid   --> false
    aspect_ratio := :equal
    # plot source points
    @series begin
        seriestype := :scatter
        markersize --> 5
        markercolor --> :red
        markershape := :circle
        WavePropBase.IO.PlotPoints(),tree.points[tree.loc2glob[tree.loc_idxs]]
    end
    # plot boudning box
    @series begin
        linecolor := :black
        linewidth := 4
        tree.bounding_box
    end
    # plot near_list
    for target in tree.data.near_list
        @series begin
            linecolor := :green
            markercolor := :green
            target.bounding_box
        end
    end
    # plot far list
    for target in tree.data.far_list
        @series begin
            linecolor := :blue
            markercolor := :blue
            target.bounding_box
        end
    end
    # plot interpolation points
    @series begin
        PlotInterpolationPoints(),tree
    end
end
