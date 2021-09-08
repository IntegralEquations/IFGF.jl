const NodesAndVals{N,Td,T} = Tuple{Array{SVector{N,Td},N},Array{T,N}}

mutable struct SourceTreeData{N,Td,T,U}
    # mesh of interpolation domain
    cart_mesh::UniformCartesianMesh{N,Td}
    # active cone indices
    active_cone_idxs::Set{CartesianIndex{N}}
    # mapping from an element `I` in the `cart_mesh` to the cheb nodes and values
    interp_data::Dict{CartesianIndex{N},NodesAndVals{N,Td,T}}
    # elements that can be interpolated
    far_list::Vector{TargetTree{N,Td,U}}
    # elements that cannot be interpolated
    near_list::Vector{TargetTree{N,Td,U}}
    # number of interpolation points per cone per dimension
    p::NTuple{N,Int}
end
function SourceTreeData{N,Td,T,U}() where {N,Td,T,U}
    if N == 2
        domain = HyperRectangle{2,Td}((0,0),(0,0))
        msh    = UniformCartesianMesh(domain,(1,1))
    elseif N == 3
        domain = HyperRectangle{3,Td}((0,0,0),(0,0,0))
        msh    = UniformCartesianMesh(domain,(1,1,1))
    else
        notimplemented()
    end
    idxs      = Set{CartesianIndex{N}}()
    interps   = Dict{CartesianIndex{N},NodesAndVals{N,Td,T}}()
    far_list  = Vector{TargetTree{N,Td,U}}()
    near_list = Vector{TargetTree{N,Td,U}}()
    p = ntuple(i->0,N)
    return SourceTreeData{N,Td,T,U}(msh,idxs,interps,far_list,near_list,p)
end

const SourceTree{N,Td,T,V,U} = ClusterTree{V,HyperRectangle{N,Td},SourceTreeData{N,Td,T,U}}

return_type(::SourceTree{N,Td,T}) where {N,Td,T} = T
cart_mesh(t::SourceTree)        = t.data.cart_mesh
active_cone_idxs(t::SourceTree) = t.data.active_cone_idxs
interp_data(t::SourceTree)      = t.data.interp_data
far_list(t::SourceTree)         = t.data.far_list
near_list(t::SourceTree)        = t.data.near_list

_addtofarlist!(s::SourceTree,t::TargetTree)  = push!(far_list(s),t)
_addtonearlist!(s::SourceTree,t::TargetTree) = push!(near_list(s),t)
_empty_interp_data!(t::SourceTree) = empty!(interp_data(t))
_addtoconeidxs!(s::SourceTree,idxcone) = push!(active_cone_idxs(s),idxcone)

"""
    interp2cart(s,source::SourceTree)

Map interpolation coordinate `s` into the corresponding cartesian coordinates.

Interpolation coordinates are similar `polar` (in 2d) or `spherical` (in 3d)
coordinates centered at the `center` of the  `source` box, except that the
radial coordinate `r` is replaced by `s = h/R`, where `h` is the radius of the
`source` box.
"""
function interp2cart(si,source::SourceTree{N}) where {N}
    bbox = container(source)
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
        @assert 0 ≤ θ < π "θ=$θ"
        @assert 0 ≤ ϕ < 2π
        r      = h/s
        x,y,z  = sph2cart(r,θ,ϕ-π)
        return SVector(x,y,z) + xc
    else
        notimplemented()
    end
end

"""
    cart2interp(x,source::SourceTree)

Map cartesian coordinates `x` into the (local) interpolation coordinates `s`.
"""
function cart2interp(x,source::SourceTree{N}) where {N}
    bbox = container(source)
    xp   = x - center(bbox)
    h    = radius(bbox)
    if N == 2
        r,θ  = cart2pol(xp...)
        s    = h/r
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
    initialize_source_tree(;points,splitter)

Create the tree-structure for clustering the source `points` using the splitting
strategy of `splitter`, returning a `SourceTree` with an empty `data` field
(i.e. `data = SourceTreeData()`).
"""
function initialize_source_tree(;points::Vector{V},splitter,datatype,Xdatatype) where V
    coords_datatype = points |> first |> coords |> typeof
    @assert coords_datatype <: SVector
    N = length(coords_datatype)
    Td = eltype(coords_datatype)
    source_tree_datatype = SourceTreeData{N,Td,datatype,Xdatatype}
    source_tree = ClusterTree{source_tree_datatype}(points,splitter)
    @assert source_tree isa SourceTree{N,Td,datatype}
    return source_tree
end

"""
    compute_interaction_list!(target_tree,source_tree,adm)

For each node in `source_tree`, compute its `near_list` and `far_list`. The
`near_list` contains all nodes in `target_tree` for which the interaction must be
computed using the direct summation.  The `far_list` includes all nodes in
`target_tree` for which an accurate interpolant in available in the source node.

The `adm` function is called through `adm(target,source)` to determine if
`target` is in the `far_list` of source. When that is not the case, we recurse
on the children of both `target` and `source` (unless both are a leaves, in which
case `target` is added to the `near_list` of source).
"""
function compute_interaction_list!(target::TargetTree,
                                   source::SourceTree,
                                   adm=admissible)
    # if either box is empty return immediately
    if length(source) == 0 || length(target) == 0
        return nothing
    end
    # handle the various possibilities
    if adm(target,source)
        _addtofarlist!(source,target)
    elseif isleaf(target) && isleaf(source)
        _addtonearlist!(source,target)
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

function admissible(target::TargetTree{N},source::SourceTree{N},η=sqrt(N)/N) where {N}
    # compute distance between source center and target box
    xc   = source |> container |> center
    h    = source |> container |> radius
    bbox = container(target)
    dc   = distance(xc,bbox)
    # if target box is outside a sphere of radius h/η, consider it admissible.
    # The same parameter η is used to construct the interpolation domain where
    # `s=h/r ∈ (0,η)` with η<1. This assures that target points which must be
    # interpolated are inside the interpolation domain.
    return dc > h/η
end

function ds_oscillatory(source,k)
    N    = ambient_dimension(source)
    h    = radius(source.bounding_box)
    return 1/(k*h) * @SVector ones(N)
end

function initialize_cone_interpolant!(source::SourceTree,p_func,ds_func)
    length(source) == 0 && (return source)
    ds      = ds_func(source)
    p       = p_func(source)
    data    = source.data
    data.p  = p
    # find minimal axis-aligned bounding box for interpolation points
    domain         = compute_interpolation_domain(source)
    all(low_corner(domain) .< high_corner(domain)) || (return source)
    data.cart_mesh = UniformCartesianMesh(domain;step=ds)
    # initialize all cones needed to cover far field
    for far_target in far_list(source) 
        for x in points(far_target) # target points
            idxcone  = cone_index(x,source)
            idxcone ∈ active_cone_idxs(source) && continue
            _addtoconeidxs!(source,idxcone)
        end
    end
    # if not a root, also create all cones needed to cover interpolation points of parents
    if !isroot(source)
        source_parent = parent(source)
        els_parent = ElementIterator(source_parent.data.cart_mesh)
        for I in active_cone_idxs(source_parent)
            rec  = els_parent[I]
            cheb = cheb2nodes_iter(data.p,rec)
            for si in cheb # interpolation node in interpolation space
                # interpolation node in physical space
                xi = interp2cart(si,source_parent)
                # mesh index for interpolation point
                idxcone = cone_index(xi,source)
                # initialize cone if needed
                idxcone ∈ active_cone_idxs(source) && continue
                _addtoconeidxs!(source,idxcone)
            end
        end
    end
    return source
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
        initialize_cone_interpolant!(source,p,ds_func)
    end
    return tree
end

"""
    compute_interpolation_domain(source::SourceTree)

Compute a domain (as a `HyperRectangle`) in interpolation space for which
`source` must provide a covering through its cone interpolants.
"""
function compute_interpolation_domain(source::SourceTree{N,Td}) where {N,Td}
    data = source.data
    lb = svector(i->typemax(Td),N) # lower bound
    ub = svector(i->typemin(Td),N) # upper bound
    # nodes on far_list
    for far_target in far_list(source)
        for x in points(far_target) # target points
            s  = cart2interp(x,source)
            lb = min.(lb,s)
            ub = max.(ub,s)
        end
    end
    if !isroot(source)
        source_parent = parent(source)
        els_parent    = ElementIterator(source_parent.data.cart_mesh)
        for idxinterp in active_cone_idxs(source_parent)
            rec = els_parent[idxinterp]
            cheb = cheb2nodes_iter(data.p,rec)
            for si in cheb
                # interpolation node in physical space
                xi = interp2cart(si,source_parent)
                s  = cart2interp(xi,source)
                lb = min.(lb,s)
                ub = max.(ub,s)
            end
        end
    end
    # create the interpolation domain
    domain = HyperRectangle(lb,ub)
    return domain
end

function cone_index(x,tree::SourceTree)
    N = ambient_dimension(tree)
    msh = tree.data.cart_mesh
    s   = cart2interp(x,tree)
    els = ElementIterator(msh)
    sz  = size(els)
    Δs  = step(msh)
    N   = length(s)
    lc  = svector(i->msh.grids[i].start,N)
    uc  = svector(i->msh.grids[i].stop,N)
    I  = ntuple(N) do n
        if s[n] ≈ lc[n]
            i = 1
        elseif s[n] ≈ uc[n]
            i = sz[n]
        else
            q = (s[n]-lc[n])/Δs[n]
            i = ceil(Int,q)
        end
        i
    end
    return CartesianIndex(I)
end

"""
    interpolation_points(s::SourceTree,I::CartesianIndex)

Return the interpolation points, in cartesian coordinates, for the cone
interpolant of index `I`.
"""
function interpolation_points(source::SourceTree,p,I::CartesianIndex)
    data   = source.data
    els    = ElementIterator(data.cart_mesh)
    rec    = els[I]
    cheb   = cheb2nodes_iter(data.p,rec)
    return map(si -> interp2cart(si,source),cheb)
end

## Plot recipes

struct PlotInterpolationPoints end

@recipe function f(::PlotInterpolationPoints,tree::SourceTree)
    for (I,interp) in interp_data(tree)
        @series begin
            seriestype := :scatter
            markercolor --> :black
            markershape --> :cross
            WavePropBase.IO.PlotPoints(), vec(interpolation_points(tree,I))
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
        WavePropBase.IO.PlotPoints(),[coords(el) for el in elements(tree)]
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
