mutable struct SourceTreeData{N,Td,U,Tv}
    # mesh of cone interpolation domain in parameter space (s,θ,ϕ)
    cone_interp_msh::UniformCartesianMesh{N,Td}
    # map from the cartesian index of active cones to a linear index.
    _cart_to_linear::Dict{CartesianIndex{N},Int}
    # elements that can be interpolated
    far_list::Vector{TargetTree{N,Td,U}}
    # elements that cannot be interpolated
    near_list::Vector{TargetTree{N,Td,U}}
    # buffer used to store interpolation data associated with node.
    _buffer::Vector{Tv}
    #
    faridxs::Dict{CartesianIndex{N},Vector{Int}}
end
function SourceTreeData{N,Td,U,Tv}() where {N,Td,U,Tv}
    if N == 2
        domain = HyperRectangle{2,Td}((0,0),(0,0))
        msh    = UniformCartesianMesh(domain,(1,1))
    elseif N == 3
        domain = HyperRectangle{3,Td}((0,0,0),(0,0,0))
        msh    = UniformCartesianMesh(domain,(1,1,1))
    else
        notimplemented()
    end
    cart2linear      = Dict{CartesianIndex{N},Int}()
    far_list         = Vector{TargetTree{N,Td,U}}()
    near_list = Vector{TargetTree{N,Td,U}}()
    buffer = Tv[]
    faridxs = Dict{CartesianIndex{N},Vector{Int}}()
    return SourceTreeData{N,Td,U,Tv}(msh,cart2linear,far_list,near_list,buffer,faridxs)
end

const SourceTree{N,Td,V,U,Tv} = ClusterTree{V,HyperRectangle{N,Td},SourceTreeData{N,Td,U,Tv}}

function allocate_buffer!(node::SourceTree,p)
    Tv = density_type(node)
    node.data._buffer = zeros(Tv,prod(p)*length(active_cone_idxs(node)))
end

function free_buffer!(node::SourceTree)
    Tv = density_type(node)
    node.data._buffer = Tv[]
end

function getbuffer(node::SourceTree,I::CartesianIndex,P)
    i = linear_cone_index(node,I)
    @views node.data._buffer[prod(P)*(i-1)+1:prod(P)*i]
end

function linear_cone_index(node::SourceTree,I::CartesianIndex)
    node.data._cart_to_linear[I]
end

density_type(::SourceTree{N,Td,V,U,Tv}) where {N,Td,V,U,Tv} = Tv

# getters
cone_interp_msh(t::SourceTree)  = t.data.cone_interp_msh
active_cone_idxs(t::SourceTree) = keys(t.data._cart_to_linear)
far_list(t::SourceTree)         = t.data.far_list
near_list(t::SourceTree)        = t.data.near_list
center(t::SourceTree)           = t |> container |> center
cone_domains(t::SourceTree)     = cone_interp_msh(t) |> ElementIterator # iterator of HyperRectangles

"""
    active_cone_domains(t::SourceTree)

Returns an iterator of the active cone interpolation domains
(as HyperRectangles in interpolation coordinates `s`).
"""
function active_cone_domains(t::SourceTree)
    iter = cone_domains(t)
    idxs = active_cone_idxs(t)
    return (iter[i] for i in idxs)
end

"""
    active_cone_idxs_and_domains(t::SourceTree)

Returns an iterator that yields `(i,rec)`, where
`i` is an active cone index and `rec` is its respective
cone interpolation domain (as a `HyperRectangle` in interpolation
coordinates `s`).
"""
function active_cone_idxs_and_domains(t::SourceTree)
    iter = cone_domains(t)
    idxs = active_cone_idxs(t)
    return ((i,iter[i]) for i in idxs)
end

# setters
_addtofarlist!(s::SourceTree,t::TargetTree)  = push!(far_list(s),t)
_addtonearlist!(s::SourceTree,t::TargetTree) = push!(near_list(s),t)
function _addtoconeidxs!(s::SourceTree,idxcone)
    dict = s.data._cart_to_linear
    haskey(dict,idxcone) || push!(dict,idxcone=>length(dict)+1)
end

function _addidxtofarlist!(node::SourceTree,I::CartesianIndex,i)
    dict = node.data.faridxs
    if haskey(dict,I)
        push!(dict[I],i)
    else
        push!(dict,I=>[i])
    end
end

function _init_cone_interp_msh!(s::SourceTree,domain,ds)
    s.data.cone_interp_msh = UniformCartesianMesh(domain;step=ds)
    return nothing
end

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
    initialize_source_tree(;points,datatype,splitter,Xdatatype)

Create the tree-structure for clustering the source `points` using the splitting
strategy of `splitter`, returning a `SourceTree` with an empty `data` field
(i.e. `data = SourceTreeData()`).

# Arguments
- `points`: vector of source points.
- `splitter`: splitting strategy.
- `Xdatatype`: type container of target points.
- `Vdatatype`: type of value stored at interpolation nodes
"""
function initialize_source_tree(;Ypoints,splitter,Xdatatype,Vdatatype)
    coords_datatype = Ypoints |> first |> coords |> typeof
    @assert coords_datatype <: SVector
    @assert isconcretetype(Xdatatype)
    N = length(coords_datatype)
    Td = eltype(coords_datatype)
    source_tree_datatype = SourceTreeData{N,Td,Xdatatype,Vdatatype}
    source_tree = ClusterTree{source_tree_datatype}(Ypoints,splitter)
    @assert source_tree isa SourceTree{N,Td}
    return source_tree
end

"""
    compute_interaction_list!(source_tree,target_tree,adm=admissible)

For each node in `source_tree`, compute its `near_list` and `far_list`. The
`near_list` contains all nodes in `target_tree` for which the interaction must be
computed using the direct summation.  The `far_list` includes all nodes in
`target_tree` for which an accurate interpolant in available in the source node.

The `adm` function is called through `adm(target,source)` to determine if
`target` is in the `far_list` of source. When that is not the case, we recurse
on the children of both `target` and `source` (unless both are a leaves, in which
case `target` is added to the `near_list` of source).
"""
function compute_interaction_list!(source::SourceTree,
                                   target::TargetTree,
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
            compute_interaction_list!(source_child,target,adm)
        end
    elseif isleaf(source)
        # recurse on children of target
        for target_child in children(target)
            compute_interaction_list!(source,target_child,adm)
        end
    else
        # recurse on children of target and source
        for source_child in children(source)
            for target_child in children(target)
                compute_interaction_list!(source_child,target_child,adm)
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

"""
    cone_domain_size_func(k)
    cone_domain_size_func(::Nothing)

Returns the function `ds_func(s::SourceTree)` that computes the size
of the cone interpolation domains. If a wavenumber `k` is passed, then
the cone domain sizes are computed in terms of `k` and the bounding box size of `s`.
This is used in oscillatory kernels, e.g. Helmholtz. If `nothing` is passed, the
cone domain sizes are set constant. This is used in static kernels, e.g. Laplace.
"""
function cone_domain_size_func(::Nothing,ds = Float64.((1,π/2,π/2)))
    # static case (e.g. Laplace)
    function ds_func(::SourceTree{N}) where N
        N != 3 && notimplemented()
        return ds
    end
    return ds_func
end
function cone_domain_size_func(k,ds = Float64.((1,π/2,π/2)))
    # oscillatory case (e.g. Helmholtz, Maxwell)
    # k: wavenumber
    function ds_func(source::SourceTree{N}) where N
        N != 3 && notimplemented()
        bbox = IFGF.container(source)
        w    = maximum(high_corner(bbox)-low_corner(bbox))
        δ    = k*w/2
        return ds ./ max(δ,1)
    end
    return ds_func
end

function initialize_cone_interpolant!(source::SourceTree,target::TargetTree,p,ds_func)
    length(source) == 0 && (return source)
    X       = root_elements(target)
    ds      = ds_func(source)
    # find minimal axis-aligned bounding box for interpolation points
    domain         = compute_interpolation_domain(source,p)
    all(low_corner(domain) .< high_corner(domain)) || (return source)
    _init_cone_interp_msh!(source,domain,ds)
    # initialize all cones needed to cover far field.
    for far_target in far_list(source)
        for i in index_range(far_target) # target points
            idxcone = cone_index(center(X[i]),source)
            _addtoconeidxs!(source,idxcone)
            _addidxtofarlist!(source,idxcone,i)
        end
    end
    # if not a root, also create all cones needed to cover interpolation points
    # of parents
    if !isroot(source)
        source_parent = parent(source)
        for rec in active_cone_domains(source_parent) # active HyperRectangles
            cheb = cheb2nodes_iter(p,rec)
            for si in cheb # interpolation node in interpolation space
                # interpolation node in physical space
                xi = interp2cart(si,source_parent)
                # mesh index for interpolation point
                idxcone = cone_index(xi,source)
                # push to list
                _addtoconeidxs!(source,idxcone)
            end
        end
    end
    return source
end

"""
    compute_cone_list!(tree::SourceTree,p,ds_func)

For each node in `source`, compute the cones which are necessary to cover
both the target points on its `far_list` as well as the interpolation points on
its parent's cones. The arguments `p` and `ds` are used to compute
an appropriate interpolation order and meshsize in interpolation space.

The algorithm starts at the root of the `source` tree, and travels down to the
leaves.
"""
function compute_cone_list!(tree::SourceTree,target,p,ds_func)
    for source in AbstractTrees.PreOrderDFS(tree)
        initialize_cone_interpolant!(source,target,p,ds_func)
    end
    return tree
end

"""
    compute_interpolation_domain(source::SourceTree)

Compute a domain (as a `HyperRectangle`) in interpolation space for which
`source` must provide a covering through its cone interpolants.
"""
function compute_interpolation_domain(source::SourceTree{N,Td},p) where {N,Td}
    lb = svector(i->typemax(Td),N) # lower bound
    ub = svector(i->typemin(Td),N) # upper bound
    # nodes on far_list
    for far_target in far_list(source)
        for el in Trees.elements(far_target) # target points
            s  = cart2interp(center(el),source)
            lb = min.(lb,s)
            ub = max.(ub,s)
        end
    end
    if !isroot(source)
        source_parent = parent(source)
        for rec in active_cone_domains(source_parent)
            cheb = cheb2nodes_iter(p,rec)
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
    msh = cone_interp_msh(tree)  # UniformCartesianMesh
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
function interpolation_points(source::SourceTree,I::CartesianIndex,p)
    data   = source.data
    els    = cone_domains(els)
    rec    = els[I]
    cheb   = cheb2nodes_iter(p,rec)
    return map(si -> interp2cart(si,source),cheb)
end

#=
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
=#
