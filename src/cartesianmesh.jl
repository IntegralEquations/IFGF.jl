"""
    struct UniformCartesianMesh{N,T} <: AbstractMesh{N,T}

An `N`-dimensional cartesian grid given as the tensor-product of `N`
one-dimensional `LinRange{T}` grids.

Iterating over a `UniformCartesianMesh` generates the elements which compose the mesh;
i.e. the `HyperRectangle` cells.
"""
struct UniformCartesianMesh{N,T}
    domain::HyperRectangle{N,T}  # stores the `low_corner` and the `high_corner` of the `UniformCartesianMesh`
    sz::NTuple{N,Int}            # number of `HyperRectangle` cells per dimension
end

low_corner(g::UniformCartesianMesh) = low_corner(g.domain)
high_corner(g::UniformCartesianMesh) = high_corner(g.domain)
domain(g::UniformCartesianMesh) = g.domain

Base.size(g::UniformCartesianMesh) = g.sz
Base.size(g::UniformCartesianMesh, i) = g.sz[i]

function grids(g::UniformCartesianMesh{N}) where {N}
    lc = low_corner(g)
    hc = high_corner(g)
    len = size(g) .+ 1   # need n+1 points for n cells
    return ntuple(N) do i
        return LinRange(lc[i], hc[i], len[i])
    end
end
function grids(g::UniformCartesianMesh, dim)
    start = low_corner(g)[dim]
    stop = high_corner(g)[dim]
    len = size(g, dim) + 1   # need n+1 points for n cells
    return LinRange(start, stop, len)
end

Base.keys(::UniformCartesianMesh{N,T}) where {N,T} = (HyperRectangle{N,T},)

function Base.step(m::UniformCartesianMesh{N}) where {N}
    lc = low_corner(m)
    hc = high_corner(m)
    sz = size(m)
    return ntuple(N) do i
        return (hc[i] - lc[i]) / sz[i]
    end
end
Base.step(m::UniformCartesianMesh, i) = step(m)[i]

"""
    UniformCartesianMesh(domain::HyperRectangle,sz::NTuple)
    UniformCartesianMesh(domain::HyperRectangle;step::NTuple)

Construct a uniform `UniformCartesianMesh` with `sz[d]` elements along dimension
`d`. If the kwarg `step` is passed, construct a `UniformCartesianMesh` with
elements of approximate size `step`.
"""
function UniformCartesianMesh(domain::HyperRectangle{N,T}, sz::Int) where {N,T}
    return UniformCartesianMesh(domain, ntuple(i -> sz, N))
end

function UniformCartesianMesh(grids::NTuple{N,LinRange{T}}) where {N,T}
    lc = SVector{N,T}(minimum(r) for r in grids)
    hc = SVector{N,T}(maximum(r) for r in grids)
    sz = length.(grids) .- 1   # need n+1 points for n cells
    domain = HyperRectangle(lc, hc)
    return UniformCartesianMesh(domain, sz)
end

# in case you pass arguments like UniformCartesianMesh(xgrid,ygrid), convert
# them to a Tuple
function UniformCartesianMesh(grids::Vararg{LinRange{T}}) where {T}
    return UniformCartesianMesh(Tuple(grids))
end

function UniformCartesianMesh(domain::HyperRectangle{N}; step) where {N}
    msg = "`step` argument must be either a `Number` or an `NTuple{$N,<:Number}`"
    step = step isa Number ? ntuple(i -> step, N) : step isa NTuple{N} ? step : error(msg)
    lc = low_corner(domain)
    hc = high_corner(domain)
    sz = ntuple(N) do i
        return Int(ceil((hc[i] - lc[i]) / step[i]))
    end
    return UniformCartesianMesh(domain, sz)
end

ambient_dimension(::UniformCartesianMesh{N}) where {N} = N

xgrid(g::UniformCartesianMesh) = grids(g, 1)
ygrid(g::UniformCartesianMesh) = grids(g, 2)
zgrid(g::UniformCartesianMesh) = grids(g, 3)

# implement ElementIterator interface to UniformCartesianMesh
function Base.CartesianIndices(m::UniformCartesianMesh)
    return CartesianIndices(size(m))
end

function Base.getindex(m::UniformCartesianMesh, I::CartesianIndex)
    N = ambient_dimension(m)
    @assert N == length(I)
    _grids = grids(m)
    lc = ntuple(N) do dim
        i = I[dim]
        return _grids[dim][i]
    end
    hc = ntuple(N) do dim
        i = I[dim] + 1
        return _grids[dim][i]
    end
    return HyperRectangle(lc, hc)
end
function Base.getindex(m::UniformCartesianMesh, I...)
    return m[CartesianIndex(I)]
end
function Base.getindex(m::UniformCartesianMesh,i::Int)
    I = CartesianIndices(iter)
    return m[I[i]]
end

"""
    element_index_for_point(p::SVector,m::UniformCartesianMesh)

Given a point `p`, return the index `I` of the element in `m` containing
`p`.
"""
function element_index_for_point(s::SVector{N}, m::UniformCartesianMesh{N}) where {N}
    els = ElementIterator(m)
    sz = size(els)
    Δs = step(m)
    lc = low_corner(m)
    uc = high_corner(m)
    # assert that lc <= s <= uc?
    @assert all(lc .<= s .<= uc) "Point $s is not inside $m"
    I = ntuple(N) do n
        q = (s[n] - lc[n]) / Δs[n]
        i = ceil(Int, q)
        return clamp(i, 1, sz[n])
    end
    return CartesianIndex(I)
end
