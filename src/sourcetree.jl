"""
    mutable struct SourceTreeData{N,T,S}

Struct used to store data associated with a node of the `SourceTree`. The
parametric type `N` represents the ambient dimension, `T` represents the
primitive type used to represent the points (e.g. `Float32` or `Float64` for
single/double precision), and `S` is the type of node in the near and far field.

# Fields:
- `msh`: uniform cartesian mesh of the interpolation space, indexable by a
  `CartesianIndex`
- `conedatadict`: dictionary mapping a cone tag `I::CartesianIndex` to the index
  range of the interpolation points owned by that cone
- `farlist`: nodes on the target tree that can be evaluated through
    interpolation
- `nearlist`: nodes on the target tree that must be evaluated by a direct sum.
"""
mutable struct SourceTreeData{N,T,S}
    msh::UniformCartesianMesh{N,T}
    conedatadict::OrderedDict{CartesianIndex{N},UnitRange{Int64}}
    farlist::Vector{S}
    nearlist::Vector{S}
end

function SourceTreeData{N,T,S}() where {N,T,S}
    domain   = HyperRectangle{N,T}(ntuple(i -> 0, N), ntuple(i -> 0, N))
    msh      = UniformCartesianMesh(domain, ntuple(i -> 1, N))
    conedict = OrderedDict{CartesianIndex{N},UnitRange{Int64}}()
    farlist  = Vector{S}()
    nearlist = Vector{S}()
    return SourceTreeData{N,T,S}(msh, conedict, farlist, nearlist)
end

"""
    const SourceTree{T,S,D}

Type alias for a `ClusterTree` with a data field `D` of type `SourceTreeData`.
"""
const SourceTree{T,S} = ClusterTree{T,S,<:SourceTreeData}

# getters for SourceTree
msh(t::SourceTree)          = t.data.msh
conedatadict(t::SourceTree) = t.data.conedatadict
# FIXME: when getting the keys of the conedatadict, it is nice to have it as a vector
active_cone_idxs(t::SourceTree)               = conedatadict(t).keys
farlist(t::SourceTree)                        = t.data.farlist
nearlist(t::SourceTree)                       = t.data.nearlist
center(t::SourceTree)                         = t |> container |> center
cone_domain(t::SourceTree, I::CartesianIndex) = t.data.msh[I]
conedata(t::SourceTree, I::CartesianIndex)    = conedatadict(t)[I]

# setters
_addtofarlist!(s::SourceTree, t::TargetTree)  = push!(farlist(s), t)
_addtonearlist!(s::SourceTree, t::TargetTree) = push!(nearlist(s), t)

"""
    _addnewcone!(node::SourceTree,I::CartesianIndex[, lock])

If `node` does not contain cone `I`, add it to the `conedict`. Does not set the
indices of the interpolation data owned by the cone.

If `lock` is passed, the operation is thread-safe.
"""
function _addnewcone!(node::SourceTree, idxcone, lck)
    cd = conedatadict(node)
    if !haskey(cd, idxcone)
        if isnothing(lck)
            push!(cd, idxcone => -1:-1)
        else
            @lock lck push!(cd, idxcone => -1:-1)
        end
    end
    return node
end

"""
    _init_msh!(node,domain,ds)

Create a uniform cartesian mesh of the interpolation domain of `node` with
element spacing `ds`.
"""
function _init_msh!(node::SourceTree, domain, ds)
    node.data.msh = UniformCartesianMesh(domain; step = ds)
    return node
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
function interp2cart(si, source::SourceTree)
    bbox = container(source)
    xc   = center(bbox)
    h    = radius(bbox)
    return interp2cart(si, xc, h)
end

function interp2cart(si, xc::SVector{N}, h::Number) where {N}
    if N == 2
        s, θ = si
        @assert -π ≤ θ < π
        r = h / s
        sin_θ, cos_θ = sincos(θ)
        x, y = r * cos(θ), r * sin(θ)
        return SVector(x, y) + xc
    elseif N == 3
        s, θ, ϕ      = si
        r            = h / s
        sin_ϕ, cos_ϕ = sincos(ϕ)
        sin_θ, cos_θ = sincos(θ)
        x            = r * cos_ϕ * sin_θ
        y            = r * sin_ϕ * sin_θ
        z            = r * cos_θ
        return SVector(x, y, z) + xc
    else
        notimplemented()
    end
end

"""
    cart2interp(x,c,h)
    cart2interp(x,source::SourceTree)

Map cartesian coordinates `x` into interpolation coordinates `s` centered at `c`
and scaled by `h`. Interpolation coordinates are similar to hyperspherical
coordinates (e.g. polar in two-dimensions and spherical in three-dimensions),
except that the radial coordinates is replaced by `s=h/r`.

If passed a `SourceTree`, use its `center` and radius as `c` and `h` respectively.
"""
@inline function cart2interp(x, source::SourceTree)
    bbox = container(source)
    c    = center(bbox)
    h    = radius(bbox)
    return cart2interp(x, c, h)
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
        θ = atan(sqrt(a), z)
        r = sqrt(a + z^2)
        s = h / r
        # s = h * invsqrt(a + z^2)
        return SVector(s, θ, ϕ)
    else
        notimplemented()
    end
end

"""
    interpolation_domain(source::SourceTree)

Compute the domain (as a `HyperRectangle`) in interpolation space for which
`source` must provide a covering through its cone interpolants.
"""
function interpolation_domain(source::SourceTree, ::Val{P}) where {P}
    N = ambient_dimension(source)
    Td = float_type(source)
    lb = svector(i -> typemax(Td), N) # lower bound
    ub = svector(i -> typemin(Td), N) # upper bound
    # nodes on far_list
    for far_target in farlist(source)
        @usethreads for x in elements(far_target) # target points
            s  = cart2interp(x, source)
            lb = min.(lb, s)
            ub = max.(ub, s)
        end
    end
    refnodes = chebnodes(P, Td)
    if !isroot(source)
        source_parent = parent(source)
        @usethreads for I in active_cone_idxs(source_parent)
            rec    = cone_domain(source_parent, I)
            lc, hc = low_corner(rec), high_corner(rec)
            c0     = (lc + hc) / 2
            c1     = (hc - lc) / 2
            for ŝ in refnodes
                si = c0 + c1 .* ŝ
                # interpolation node in physical space
                xi = interp2cart(si, source_parent)
                s  = cart2interp(xi, source)
                lb = min.(lb, s)
                ub = max.(ub, s)
            end
        end
    end
    # create the interpolation domain
    domain = HyperRectangle(lb, ub)
    return domain
end

"""
    cone_index(x::SVector,tree::SourceTree)

Given a point `x` in cartesian coordinates, return the index `I` of the cone in
`source` to which `x` belongs, as well as its parametric coordinate `s`.

## See also: [`cart2interp`](@ref)
"""
function cone_index(x::SVector, tree::SourceTree)
    m = msh(tree)  # UniformCartesianMesh
    s = cart2interp(x, tree)
    I = element_index(s, m)
    return I, s
end

"""
    element_index(s::SVector,m::UniformCartesianMesh)

Given a point `s ∈ m`, return the index `I` of the element in `m` containing
`s`.
"""
function element_index(s::SVector{N}, m::UniformCartesianMesh{N}) where {N}
    sz = size(m)
    Δs = step(m)
    lc = low_corner(m)
    uc = high_corner(m)
    # assert that lc <= s <= uc?
    # @assert all(lc .<= s .<= uc)
    I = ntuple(N) do n
        q = (s[n] - lc[n]) / Δs[n]
        i = ceil(Int, q)
        return clamp(i, 1, sz[n])
    end
    return CartesianIndex(I)
end
