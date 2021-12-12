const ConeData{N,T, I} = @NamedTuple{irange::UnitRange{I},
                                    faridxs::Vector{I},
                                    paridxs::Vector{I},
                                    farpcoords::Vector{SVector{N,T}},
                                    parpcoords::Vector{SVector{N,T}}}

function ConeData{N,T,I}(irange::UnitRange) where {N,T,I}
    ConeData{N,T,I}((irange=irange,faridxs=I[],paridxs=I[],farpcoords=SVector{N,T}[],parpcoords=SVector{N,T}[]))
end

mutable struct SourceTreeData{N,Td,U,Tv}
    # mesh of cone interpolation domain in parameter space (s,θ,ϕ)
    cone_interp_msh::UniformCartesianMesh{N,Td}
    # data associated with each interpolant
    conedata::Dict{CartesianIndex{N},ConeData{N,Td,Int}}
    # map from the cartesian index of active cones to a linear index.
    _cart_to_linear::Dict{CartesianIndex{N},Int}
    # elements that can be interpolated
    far_list::Vector{TargetTree{N,Td,U}}
    # elements that cannot be interpolated
    near_list::Vector{TargetTree{N,Td,U}}
    # buffer used to store interpolation values
    ivals::Vector{Tv}
    # buffer used to store interpolation nodes
    ipts::Vector{SVector{N,Td}}
    faridxs::Dict{CartesianIndex{N},Vector{Int}}
    paridxs::Dict{CartesianIndex{N},Vector{Int}}
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
    conedata = Dict{CartesianIndex{N},ConeData{N,Td,Int}}()
    cart2linear      = Dict{CartesianIndex{N},Int}()
    far_list         = Vector{TargetTree{N,Td,U}}()
    near_list = Vector{TargetTree{N,Td,U}}()
    ivals = Tv[]
    ipts = SVector{N,Td}[]
    faridxs = Dict{CartesianIndex{N},Vector{Int}}()
    paridxs = Dict{CartesianIndex{N},Vector{Int}}()
    return SourceTreeData{N,Td,U,Tv}(msh,conedata,cart2linear,far_list,near_list,ivals,ipts,faridxs,paridxs)
end

const SourceTree{N,Td,V,U,Tv} = ClusterTree{V,HyperRectangle{N,Td},SourceTreeData{N,Td,U,Tv}}

density_type(::SourceTree{N,Td,V,U,Tv}) where {N,Td,V,U,Tv} = Tv

function allocate_buffer!(node::SourceTree,p)
    Tv = density_type(node)
    node.data.ivals = zeros(Tv,prod(p)*length(active_cone_idxs(node)))
end

function free_buffer!(node::SourceTree)
    Tv = density_type(node)
    node.data.ivals = Tv[]
end

function getbuffer(node::SourceTree,I::CartesianIndex)
    idxrange = coneidxrange(node,I)
    @views node.data.ivals[idxrange]
end

function interp_points(node,I)
    idxrange = coneidxrange(node,I)
    @views node.data.ipts[idxrange]
end

function coneidxrange(node::SourceTree,I::CartesianIndex)
    conedata(node)[I].irange
end

function linear_cone_index(node::SourceTree,I::CartesianIndex)
    node.data._cart_to_linear[I]
end

# getters
cone_interp_msh(t::SourceTree)  = t.data.cone_interp_msh
conedata(t::SourceTree)                    = t.data.conedata
faridxs(t::SourceTree,I::CartesianIndex) =  conedata(t)[I].faridxs
paridxs(t::SourceTree,I::CartesianIndex) =  conedata(t)[I].paridxs
active_cone_idxs(t::SourceTree) = keys(conedata(t))
far_list(t::SourceTree)         = t.data.far_list
near_list(t::SourceTree)        = t.data.near_list
center(t::SourceTree)           = t |> container |> center
cone_domains(t::SourceTree)     = cone_interp_msh(t) |> ElementIterator # iterator of HyperRectangles
cone_domain(t::SourceTree,I::CartesianIndex) = cone_domains(t)[I]

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

function _addtoconeidxs!(node::SourceTree{N,Td},idxcone,p) where {N,Td}
    dict      = node.data._cart_to_linear
    coneid   = length(dict) + 1
    haskey(dict,idxcone) || push!(dict,idxcone=>coneid)
    ##
    cd = conedata(node)
    if !haskey(cd,idxcone)
        coneid   = length(cd) + 1
        irange = prod(p)*(coneid-1)+1:(coneid)*prod(p)
        push!(cd,idxcone=>ConeData{N,Td,Int}(irange))
        rec = cone_domain(node,idxcone)
        s   = cheb2nodes_iter(p,rec) # cheb points in parameter space
        x   = map(si->interp2cart(si,node),s) |> vec |> collect # cheb points in physical space
        ipts = node.data.ipts
        append!(ipts,x)
    end
end

function _addidxtofarlist!(node::SourceTree,I::CartesianIndex,i)
    idxs = faridxs(node,I)
    push!(idxs,i)
end

function _addidxtoparlist!(node::SourceTree,I::CartesianIndex,i::Int)
    idxs = paridxs(node,I)
    push!(idxs,i)
    dict = node.data.paridxs
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
    interp2cart(s,xc,h)
    interp2cart(s,source::SourceTree)

Map interpolation coordinate `s` into the corresponding cartesian coordinates.

Interpolation coordinates are similar to `polar` (in 2d) or `spherical` (in 3d)
coordinates centered at the `xc`, except that the
radial coordinate `r` is replaced by `s = h/r`.

If passed a `SourceTree` as second argument, call `interp2cart(s,xc,h)` with
`xc` the center of the `SourceTree` and `h` its radius.
"""
function interp2cart(si,source::SourceTree{N}) where {N}
    bbox = container(source)
    xc   = center(bbox)
    h    = radius(bbox)
    interp2cart(si,xc,h)
end

function interp2cart(si,xc::SVector{N},h::Number) where {N}
    if N == 2
        s,θ  = si
        @assert -π ≤ θ < π
        r    = h/s
        x    = r*cos(θ)
        y    = r*sin(θ)
        return SVector(x,y) + xc
    elseif N == 3
        s,θ,ϕ  = si
        @assert 0 ≤ θ < π "θ=$θ"
        @assert -π ≤ ϕ < π
        r      = h/s
        x = r*cos(ϕ)*sin(θ)
        y = r*sin(ϕ)*sin(θ)
        z = r*cos(θ)
        return SVector(x,y,z) + xc
    else
        notimplemented()
    end
end

"""
    cart2interp(x,source::SourceTree)

Map cartesian coordinates `x` into the (local) interpolation coordinates `s`.
"""
@inline function cart2interp(x,source::SourceTree{N}) where {N}
    bbox = container(source)
    c    = center(bbox)
    h    = radius(bbox)
    cart2interp(x,c,h)
end

@inline function cart2interp(x,c::SVector{N},h) where {N}
    xp = x - c
    if N == 2
        x,y = xp
        r = sqrt(x^2 + y^2)
        θ = atan(y,x)
        s = h/r
        return SVector(s,θ)
    elseif N == 3
        x,y,z = xp
        ϕ     = atan(y,x)
        a     = x^2 + y^2
        θ     = atan(sqrt(a),z)
        r     = sqrt(a + z^2)
        s     = h/r
        return SVector(s,θ,ϕ)
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
    N  = length(coords_datatype)
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
            _addtoconeidxs!(source,idxcone,p)
            _addidxtofarlist!(source,idxcone,i)
        end
    end
    # if not a root, also create all cones needed to cover interpolation points
    # of parents
    if !isroot(source)
        source_parent = parent(source)
        ipts = source_parent.data.ipts
        for (i,x) in enumerate(ipts)
            idxcone = cone_index(x,source)
            _addtoconeidxs!(source,idxcone,p)
            _addidxtoparlist!(source,idxcone,i)
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
        allocate_buffer!(source,p)
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

function cone_index(x::SVector{N},tree::SourceTree{N}) where {N}
    msh = cone_interp_msh(tree)  # UniformCartesianMesh
    s   = cart2interp(x,tree)
    els = ElementIterator(msh)
    sz  = size(els)
    Δs  = step(msh)
    lc  = svector(i->msh.grids[i].start,N)
    uc  = svector(i->msh.grids[i].stop,N)
    # assert that lc <= s <= uc?
    @assert all( lc .<= s .<= uc)
    I  = ntuple(N) do n
        q = (s[n]-lc[n]) / Δs[n]
        i = ceil(Int,q)
        clamp(i,1,sz[n])
    end
    return CartesianIndex(I)
end

# using LoopVectorization
# function cone_index(X,tree)
#     msh = cone_interp_msh(tree)  # UniformCartesianMesh
#     els = ElementIterator(msh)
#     sz1,sz2,sz3  = size(els)
#     ds1,ds2,ds3  = step(msh)
#     lc  = svector(i->msh.grids[i].start,3)
#     bbox = container(tree)
#     c    = center(bbox)
#     h    = radius(bbox)
#     out  = Vector{CartesianIndex{3}}(undef,length(X))
#     _out = reinterpret(reshape,Int,out)
#     _X   = reinterpret(reshape,Float64,X)
#     @turbo for i in 1:length(X)
#         x = _X[1,i] - c[1]
#         y = _X[2,i] - c[2]
#         z = _X[3,i] - c[3]
#         ϕ     = atan(y,x)
#         a     = x^2 + y^2
#         θ     = atan(sqrt(a),z)
#         r     = sqrt(a + z^2)
#         s     = h/r
#         q = (s-lc[1]) ÷ ds1 |> Int
#         _out[1,i] = min(max(q,1),sz1)
#         q = (θ-lc[2]) ÷ ds2 |> Int
#         _out[2,i] = min(max(q,1),sz2)
#         q = (ϕ-lc[3]) ÷ ds3 |> Int
#         _out[3,i] = min(max(q,1),sz3)
#     end
#     return out
# end

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
