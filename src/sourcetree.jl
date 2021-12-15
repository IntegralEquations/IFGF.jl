"""
    const ConeData{N,T,I}

`NamedTuple` used to store data associated with a single cone in the [`SourceTree`](@ref)
structure. Nodes in the `SourceTree{N,T}` contain a `conedatadict` mapping indices of
its active cones to a `ConeData` instance. The type parameter `I` is the integer
type used to store indices (defaults to `Int`).

# Fields:
- `irange`:  range of indices in `ivals` used for the cone interpolant
- `faridxs`: vector of indices in the far field covered by the cone
- `paridxs`: vector of indices in `ivals` of the parent covered by the cone
- `farpcoords`: local parametric coordinates corresponding to each point in
  `faridxs`
- `parpcoords`: local parametric coordinates corresponding to each point in `paridxs`
"""
const ConeData{N,T,I} = @NamedTuple {
    irange::UnitRange{I},
    faridxs::Vector{I},
    paridxs::Vector{I},
    farpcoords::Vector{SVector{N,T}},
    parpcoords::Vector{SVector{N,T}}}

function ConeData{N,T,I}(irange::UnitRange) where {N,T,I}
    ConeData{N,T,I}((irange = irange, faridxs = I[], paridxs = I[], farpcoords = SVector{N,T}[], parpcoords = SVector{N,T}[]))
end

"""
    mutable struct SourceTreeData{N,T,V}

Struct used to store data associated with a node of the `SourceTree`. The
parametric type `N` represents the ambient dimension, `T` represents the
primitive type used to represent the points (e.g. `Float64` or `Float32` for
single/double precision), and `V` is the type of the density that will be
interpolated.

# Fields:
- `msh`: uniform cartesian mesh of the interpolation space,
  indexable by a `CartesianIndex`
- `conedatadict`: dictionary mapping a cone tag `I::CartesianIndex` to its
  [`ConeData`](@ref)
- `farlist`: nodes on the target tree that can be evaluated through
  interpolation
- `nearlist`: nodes on the target tree that must be evaluated by a direct sum.
- `ivals`: vector of cone interpolant values/coefficients.
- `ipts`: vector of interpolation nodes in cartesian coordiantes.
"""
mutable struct SourceTreeData{N,T,V}
    msh::UniformCartesianMesh{N,T}
    conedatadict::Dict{CartesianIndex{N},ConeData{N,T,Int}}
    farlist::Vector{TargetTree{N,T}}
    nearlist::Vector{TargetTree{N,T}}
    ivals::Vector{V}
    ipts::Vector{SVector{N,T}}
end

function SourceTreeData{N,T,V}() where {N,T,V}
    domain = HyperRectangle{N,T}(ntuple(i->0,N),ntuple(i->0,N))
    msh    = UniformCartesianMesh(domain, ntuple(i->1,N))
    conedatadict = Dict{CartesianIndex{N},ConeData{N,T,Int}}()
    farlist = Vector{TargetTree{N,T}}()
    nearlist = Vector{TargetTree{N,T}}()
    ivals = V[]
    ipts = SVector{N,T}[]
    return SourceTreeData{N,T,V}(msh, conedatadict, farlist, nearlist, ivals, ipts)
end

"""
    const SourceTree{N,T,V}

Type alias for a `ClusterTree` of points (represented as `SVector{N,T}`) storing
data of type [`SourceTreeData{N,T,V}`](@ref). See the documentation of
`ClusterTree` in [`WavePropBase`](https://github.com/WaveProp/WavePropBase.jl)
for more information.
"""
const SourceTree{N,T,V} = ClusterTree{Vector{SVector{N,T}},HyperRectangle{N,T},SourceTreeData{N,T,V}}

"""
    const ConeDataLite{I}

Similar to [`ConeData`](@ref), but stores only the `irange` field.
"""
const ConeDataLite{I} = @NamedTuple {irange::UnitRange{I}}

function ConeDataLite(I::UnitRange{T}) where {T}
    ConeDataLite{T}((I,))
end

"""
    mutable struct SourceTreeDataLite{N,T,V}

Similar to [`SourceTreeData`](@ref), but uses the [`ConeDataLite`](@ref)
structure to store the cone data.
"""
mutable struct SourceTreeDataLite{N,T,V}
    msh::UniformCartesianMesh{N,T}
    conedatadict::Dict{CartesianIndex{N},ConeDataLite{Int}}
    farlist::Vector{TargetTree{N,T}}
    nearlist::Vector{TargetTree{N,T}}
    ivals::Vector{V}
end

function SourceTreeDataLite{N,T,V}() where {N,T,V}
    domain       = HyperRectangle{N,T}(ntuple(i->0,N),ntuple(i->0,N))
    msh          = UniformCartesianMesh(domain, ntuple(i->1,N))
    conedatadict = Dict{CartesianIndex{N},ConeDataLite{Int}}()
    farlist      = Vector{TargetTree{N,T}}()
    nearlist     = Vector{TargetTree{N,T}}()
    ivals        = V[]
    return SourceTreeDataLite{N,T,V}(msh, conedatadict, farlist, nearlist, ivals)
end

"""
    const SourceTreeLite{N,T,V}

Type alias for a `ClusterTree` of points (represented as `SVector{N,T}`) storing
data of type [`SourceTreeDataLite{N,T,V}`](@ref). See the documentation of
`ClusterTree` in [`WavePropBase`](https://github.com/WaveProp/WavePropBase.jl)
for more information.
"""
const SourceTreeLite{N,T,V} = ClusterTree{Vector{SVector{N,T}},HyperRectangle{N,T},SourceTreeDataLite{N,T,V}}

density_type(::SourceTree{N,T,V}) where {N,T,V} = V
density_type(::SourceTreeLite{N,T,V}) where {N,T,V} = V

"""
    allocate_ivals!(node,p)

Allocate the memory necessary to construct all cone interpolants of order `p`
for `node`. This assumes that the list of active cones of `node` has already
been computed.
"""
function allocate_ivals!(node::Union{SourceTree,SourceTreeLite},::Val{P}) where {P}
    V = density_type(node)
    node.data.ivals = zeros(V, prod(P) * length(active_cone_idxs(node)))
end

"""
    free_ivals!(node)

Free the memory used to store the cone interpolant values of `node`.
"""
function free_ivals!(node::Union{SourceTree,SourceTreeLite})
    empty!(node.data.ivals)
end

"""
    coneidxrange(node::SourceTree,I::CartesianIndex)

Index range of points/values corresponding to interpolation cone `I`. Used to
index the global vectors `ivals` and `ipts` in the `node`.
"""
function coneidxrange(node::SourceTree, I::CartesianIndex)
    conedata(node, I).irange
end

function coneidxrange(node::SourceTreeLite, I::CartesianIndex)
    conedata(node, I).irange
end

"""
    getivals(node::SourceTree,I::CartesianIndex)

Return a view of the interpolation values corresponding to the cone with index
`I` on `node`.
"""
function getivals(node::Union{SourceTree,SourceTreeLite}, I::CartesianIndex)
    idxrange = coneidxrange(node, I)
    @views node.data.ivals[idxrange]
end

# getters for both SourceTree and SourceTreeLite
msh(t::Union{SourceTree,SourceTreeLite})                            = t.data.msh
conedatadict(t::Union{SourceTree,SourceTreeLite})                   = t.data.conedatadict
conedata(t::Union{SourceTree,SourceTreeLite}, I::CartesianIndex)    = t.data.conedatadict[I]
active_cone_idxs(t::Union{SourceTree,SourceTreeLite})               = keys(conedatadict(t))
farlist(t::Union{SourceTree,SourceTreeLite})                        = t.data.farlist
nearlist(t::Union{SourceTree,SourceTreeLite})                       = t.data.nearlist
WavePropBase.center(t::Union{SourceTree,SourceTreeLite})            = t |> container |> center
cone_domains(t::Union{SourceTree,SourceTreeLite})                   = msh(t) |> ElementIterator # iterator of HyperRectangles
cone_domain(t::Union{SourceTree,SourceTreeLite}, I::CartesianIndex) = cone_domains(t)[I]

# getters only valid for SourceTree (and not SourceTreeLite)
faridxs(t::SourceTree, I::CartesianIndex)    = conedata(t, I).faridxs
farpcoords(t::SourceTree, I::CartesianIndex) = conedata(t, I).farpcoords
paridxs(t::SourceTree, I::CartesianIndex)    = conedata(t, I).paridxs
parpcoords(t::SourceTree, I::CartesianIndex) = conedata(t, I).parpcoords

# setters
_addtofarlist!(s::Union{SourceTree,SourceTreeLite}, t::TargetTree)  = push!(farlist(s), t)
_addtonearlist!(s::Union{SourceTree,SourceTreeLite}, t::TargetTree) = push!(nearlist(s), t)

"""
    _addnewcone!(node::SourceTree,I::CartesianIndex,p)

If `node` does not contain cone `I`, add it to the `conedatadict` and push the
new interpolation points to the `ipts` field.
"""
function _addnewcone!(node::SourceTree{N,T}, idxcone, p::Val{P}) where {N,T,P}
    cd = conedatadict(node)
    if !haskey(cd, idxcone)
        coneid = length(cd) + 1
        irange = prod(P)*(coneid-1)+1:(coneid)*prod(P)
        push!(cd, idxcone => ConeData{N,T,Int}(irange))
        rec = cone_domain(node, idxcone)
        refnodes = cheb2nodes(p)
        s = cheb2nodes(p, rec) # cheb points in parameter space
        x = map(si -> interp2cart(si, node), s) |> vec |> collect # cheb points in physical space
        ipts = node.data.ipts
        append!(ipts, x)
    end
    return node
end

function _addnewcone!(node::SourceTreeLite, idxcone, p::Val{P}) where {P}
    cd = conedatadict(node)
    if !haskey(cd, idxcone)
        coneid            = length(cd) + 1
        irange            = prod(P)*(coneid-1)+1:(coneid)*prod(P)
        push!(cd, idxcone => ConeDataLite(irange))
    end
    return node
end

"""
    _addidxstofarlist(node,I,i,s)

Add the index `i` and parametric coordinate `s` to the list of [`faridxs`](@ref)
and  [`farpcoords`](@ref) of cone `I`.
"""
@inline function _addidxtofarlist!(node::SourceTree, I::CartesianIndex, i, s)
    idxs    = faridxs(node, I)
    push!(idxs, i)
    pcoords = farpcoords(node, I)
    push!(pcoords, s)
    return node
end

"""
    _addidxstoparlist(node,I,i,s)

Add the index `i` and parametric coordinate `s` to the list of [`paridxs`](@ref)
and  [`parpcoords`](@ref) of cone `I`.
"""
function _addidxtoparlist!(node::SourceTree, I::CartesianIndex, i::Int, s::SVector)
    idxs = paridxs(node, I)
    pcoords = parpcoords(node, I)
    push!(idxs, i)
    push!(pcoords, s)
    return node
end

"""
    _init_msh!(node,domain,ds)

Create a uniform cartesian mesh of the interpolation domain of `node` with
element spacing `ds`.
"""
function _init_msh!(node::Union{SourceTree,SourceTreeLite}, domain, ds)
    node.data.msh = UniformCartesianMesh(domain; step = ds)
    return node
end

"""
    interp2cart(s,xc,h)
    interp2cart(s,source::Union{SourceTree,SourceTreeLite})

Map interpolation coordinate `s` into the corresponding cartesian coordinates.

Interpolation coordinates are similar to `polar` (in 2d) or `spherical` (in 3d)
coordinates centered at the `xc`, except that the
radial coordinate `r` is replaced by `s = h/r`.

If passed a `SourceTree` or `SourceTreeLite` as second argument, call `interp2cart(s,xc,h)` with
`xc` the center of the `SourceTree` and `h` its radius.
"""
function interp2cart(si, source::Union{SourceTree{N},SourceTreeLite{N}}) where {N}
    bbox = container(source)
    xc   = center(bbox)
    h    = radius(bbox)
    interp2cart(si, xc, h)
end

function interp2cart(si, xc::SVector{N}, h::Number) where {N}
    if N == 2
        s, θ = si
        @assert -π ≤ θ < π
        r = h / s
        sin_θ,cos_θ = sincos(θ)
        x,y = r*cos(θ), r*sin(θ)
        return SVector(x, y) + xc
    elseif N == 3
        s, θ, ϕ     = si
        r           = h / s
        sin_ϕ,cos_ϕ = sincos(ϕ)
        sin_θ,cos_θ = sincos(θ)
        x           = r * cos_ϕ * sin_θ
        y           = r * sin_ϕ * sin_θ
        z           = r * cos_θ
        return SVector(x, y, z) + xc
    else
        notimplemented()
    end
end

"""
    cart2interp(x,c,h)
    cart2interp(x,source::Union{SourceTree,SourceTreeLite})

Map cartesian coordinates `x` into interpolation coordinates `s` centered at `c`
and scaled by `h`. Interpolation coordinates are similar to hyperspherical
coordinates (e.g. polar in two-dimensions and spherical in three-dimensions),
except that the radial coordinates is replaced by `s=h/r`.

If passed a `SourceTree`, use its `center` and radius as `c` and `h` respectively.
"""
@inline function cart2interp(x, source::Union{SourceTree{N},SourceTreeLite{N}}) where {N}
    bbox = container(source)
    c    = center(bbox)
    h    = radius(bbox)
    cart2interp(x, c, h)
end

@inline function cart2interp(x, c::SVector{N}, h) where {N}
    xp = x - c
    if N == 2
        x, y = xp
        r = sqrt(x^2 + y^2)
        θ = atan(y, x)
        s = h / r
        return SVector(s, θ)
    elseif N == 3
        x, y, z = xp
        ϕ = atan(y, x)
        a = x^2 + y^2
        θ = atan(sqrt(a),z)
        r = sqrt(a + z^2)
        s = h / r
        # s = h * invsqrt(a + z^2)
        return SVector(s, θ, ϕ)
    else
        notimplemented()
    end
end

function _cart2interp3d_vec!(out,X::AbstractMatrix, h) where {N}
    m,n = size(X)
    @turbo for i in 1:n
        a        = X[1,i]^2 + X[2,i]^2
        out[1,i] = h / sqrt(a + X[3,i]^2) # s
        out[2,i] = atan(sqrt(a),X[3,i])   # θ
        out[3,i] = atan(X[2,i], X[1,i])   # ϕ
    end
    return out
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
function initialize_source_tree(; Ypoints::Vector{SVector{N,T}}, splitter, Vdatatype, lite=false) where {N,T}
    source_tree_datatype = lite ? SourceTreeDataLite{N,T,Vdatatype} : SourceTreeData{N,T,Vdatatype}
    source_tree = ClusterTree{source_tree_datatype}(Ypoints, splitter)
    return source_tree
end

"""
    compute_interpolation_domain(source::SourceTree)

Compute a domain (as a `HyperRectangle`) in interpolation space for which
`source` must provide a covering through its cone interpolants.
"""
function compute_interpolation_domain(source::SourceTree{N,T},p::Val{P}) where {N,T,P}
    lb = svector(i -> typemax(T), N) # lower bound
    ub = svector(i -> typemin(T), N) # upper bound
    # nodes on farlist
    nfar = isempty(farlist(source)) ? 0 : sum(x->length(x),farlist(source))
    farpcoords = Vector{SVector{N,T}}(undef,nfar)
    cc = 0
    for far_target in farlist(source)
        for x in Trees.elements(far_target) # target points
            cc += 1
            s = cart2interp(x, source)
            farpcoords[cc] = s
            lb = min.(lb, s)
            ub = max.(ub, s)
        end
    end
    source_parent = parent(source)
    ipts = source_parent.data.ipts
    parpcoords = Vector{SVector{N,T}}(undef,length(ipts))
    cc = 0
    if !isroot(source)
        for x in ipts
            cc += 1
            s = cart2interp(x, source)
            parpcoords[cc] = s
            lb = min.(lb, s)
            ub = max.(ub, s)
        end
    end
    # create the interpolation domain
    domain = HyperRectangle(lb, ub)
    return domain, farpcoords, parpcoords
end

function compute_interpolation_domain(source::SourceTreeLite{N,Td},p::Val{P}) where {N,Td,P}
    lb = svector(i->typemax(Td),N) # lower bound
    ub = svector(i->typemin(Td),N) # upper bound
    # nodes on far_list
    for far_target in farlist(source)
        for x in Trees.elements(far_target) # target points
            s  = cart2interp(x,source)
            lb = min.(lb,s)
            ub = max.(ub,s)
        end
    end
    refnodes = cheb2nodes(p)
    if !isroot(source)
        source_parent = parent(source)
        for I in active_cone_idxs(source_parent)
            rec  = cone_domain(source_parent,I)
            lc,hc = low_corner(rec), high_corner(rec)
            c0 = (lc+hc)/2
            c1 = (hc-lc)/2
            cheb  = map(x-> c0 + c1 .* x,refnodes)
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

function cone_index(x::SVector{N}, tree::Union{SourceTree{N},SourceTreeLite{N}}) where {N}
    m = msh(tree)  # UniformCartesianMesh
    s = cart2interp(x, tree)
    I = element_index(s,m)
    return I, s
end

function element_index(s::SVector{N},m::UniformCartesianMesh{N}) where {N}
    els = ElementIterator(m)
    sz = size(els)
    Δs = step(m)
    lc = svector(i -> m.grids[i].start, N)
    uc = svector(i -> m.grids[i].stop, N)
    # assert that lc <= s <= uc?
    # @assert all(lc .<= s .<= uc)
    I = ntuple(N) do n
        q = (s[n] - lc[n]) / Δs[n]
        i = ceil(Int, q)
        clamp(i, 1, sz[n])
    end
    return CartesianIndex(I)
end
