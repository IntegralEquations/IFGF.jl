"""
    struct HyperRectangle{N,T}

Axis-aligned hyperrectangle in `N` dimensions given by `low_corner::SVector{N,T}` and
`high_corner::SVector{N,T}`. By default, `T` is `Float64`.
"""
struct HyperRectangle{N,T}
    low_corner::SVector{N,T}
    high_corner::SVector{N,T}
end

low_corner(r::HyperRectangle) = r.low_corner
high_corner(r::HyperRectangle) = r.high_corner

"""
    HyperRectangle(lc,hc)

Construct an axis-aligned hyperrectangle with low-corner `lc` and high-corner
`hc`.

It is recommended that `lc` and `hc` be of type `SVector` or `Tuple` to avoid
type instabilities.

```jldoctest
julia> IFGF.HyperRectangle((0.0, 0.0), (1.0, 2.0))
IFGF.HyperRectangle{2, Float64}([0.0, 0.0], [1.0, 2.0])
```

"""
HyperRectangle(l::SVector, h) = HyperRectangle(promote(l, h)...)
HyperRectangle(l::Tuple, h::Tuple) = HyperRectangle(SVector(l), SVector(h))
HyperRectangle(l::Vector, h::Vector) = HyperRectangle(Tuple(l), Tuple(h))
HyperRectangle(a::Number, b::Number) = HyperRectangle(SVector(a), SVector(b)) # 1D

ambient_dimension(::HyperRectangle{N}) where {N} = N
geometric_dimension(::HyperRectangle{N}) where {N} = N

float_type(::HyperRectangle{N,T}) where {N,T} = T

"""
    diameter(Ω)

Largest distance between `x` and `y` for `x,y ∈ Ω`.
"""
diameter(r::HyperRectangle) = norm(high_corner(r) .- low_corner(r), 2)

"""
    center(Ω)

Center of the smallest ball containing `Ω`.
"""
center(r::HyperRectangle) = (low_corner(r) + high_corner(r)) / 2

"""
    radius(Ω)

Half the [`diameter`](@ref).
"""
radius(r::HyperRectangle) = diameter(r) / 2

function Base.in(point, h::HyperRectangle, tol = 0)
    return all(low_corner(h) .- tol .<= point .<= high_corner(h) .+ tol)
end

function vertices(rec::HyperRectangle{2})
    lc = low_corner(rec)
    hc = high_corner(rec)
    return SVector(
        SVector(lc[1], lc[2]),
        SVector(hc[1], lc[2]),
        SVector(hc[1], hc[2]),
        SVector(lc[1], hc[2]),
    )
end
function vertices(rec::HyperRectangle{3})
    lc = low_corner(rec)
    hc = high_corner(rec)
    return SVector(
        # lower face
        SVector(lc[1], lc[2], lc[3]),
        SVector(hc[1], lc[2], lc[3]),
        SVector(hc[1], hc[2], lc[3]),
        SVector(lc[1], hc[2], lc[3]),
        # upper face
        SVector(lc[1], lc[2], hc[3]),
        SVector(hc[1], lc[2], hc[3]),
        SVector(hc[1], hc[2], hc[3]),
        SVector(lc[1], hc[2], hc[3]),
    )
end

"""
    distance(Ω₁,Ω₂)

Minimal Euclidean distance between a point `x ∈ Ω₁` and `y ∈ Ω₂`.
"""
function distance(rec1::HyperRectangle{N}, rec2::HyperRectangle{N}) where {N}
    d2 = 0
    rec1_low_corner = low_corner(rec1)
    rec1_high_corner = high_corner(rec1)
    rec2_low_corner = low_corner(rec2)
    rec2_high_corner = high_corner(rec2)
    for i in 1:N
        d2 +=
            max(0, rec1_low_corner[i] - rec2_high_corner[i])^2 +
            max(0, rec2_low_corner[i] - rec1_high_corner[i])^2
    end
    return sqrt(d2)
end

"""
    distance(x::SVector,r::HyperRectangle)

The (minimal) Euclidean distance between the point `x` and any point `y ∈ r`.
"""
function distance(pt::SVector{N}, rec::HyperRectangle{N}) where {N}
    d2 = 0
    rec_low_corner = low_corner(rec)
    rec_high_corner = high_corner(rec)
    for i in 1:N
        d2 += max(0, pt[i] - rec_high_corner[i])^2 + max(0, rec_low_corner[i] - pt[i])^2
    end
    return sqrt(d2)
end
distance(rec::HyperRectangle{N}, pt::SVector{N}) where {N} = distance(pt, rec)

function Base.isapprox(h1::HyperRectangle, h2::HyperRectangle; kwargs...)
    return isapprox(h1.low_corner, h2.low_corner; kwargs...) &&
           isapprox(h1.high_corner, h2.high_corner; kwargs...)
end
width(r::HyperRectangle) = high_corner(r) - low_corner(r)
half_width(r::HyperRectangle) = width(r) / 2

function Base.intersect(r1::HyperRectangle, r2::HyperRectangle)
    lb = max.(low_corner(r1), low_corner(r2))
    ub = min.(high_corner(r1), high_corner(r2))
    return HyperRectangle(lb, ub)
end

"""
    union(r1::HyperRectangle,r2::HyperRectangle)

Smallest [`HyperRectangle`](@ref) that contains both `r1` and `r2`.
"""
function Base.union(r1::HyperRectangle, r2::HyperRectangle)
    lb = min.(low_corner(r1), low_corner(r2))
    ub = max.(high_corner(r1), high_corner(r2))
    return HyperRectangle(lb, ub)
end

Base.isempty(r::HyperRectangle) = any(low_corner(r) .> high_corner(r))

"""
    bounding_box(els[, cube=false])

Compute a [`HyperRectangle`](@ref) that contains the [`center`](@ref) of all elements in
`els`. If `cube==true`, the result is a square/cube instead.

Works for any type that implements `center`.
"""
function bounding_box(els, cube = false)
    isempty(els) && (error("data cannot be empty"))
    lb = center(first(els))
    ub = center(first(els))
    for el in els
        pt = center(el)
        lb = min.(lb, pt)
        ub = max.(ub, pt)
    end
    if cube # fit a square/cube instead
        w = maximum(ub - lb)
        xc = (ub + lb) / 2
        # min/max below helps avoid floating point issues with results with the
        # cube not being exactly contained in the original rectangle
        lb = min.(xc .- w / 2, lb)
        ub = max.(xc .+ w / 2, ub)
        # TODO: return HyperCube instead
    end
    lb == ub && (lb = prevfloat.(lb); ub = nextfloat.(ub)) # to avoid "empty" rectangles
    return HyperRectangle(lb, ub)
end

"""
    split(rec::HyperRectangle[, axis::Int, place])

Split a hyperrectangle in two along the `axis` direction at the  position
`place`. Returns a tuple with the two resulting hyperrectangles.

When no `place` is given, defaults to splitting in the middle of the axis.

When no axis and no place are given, defaults to splitting along the largest axis.
"""
function Base.split(rec::HyperRectangle, axis, place)
    rec_low_corner = low_corner(rec)
    rec_high_corner = high_corner(rec)
    N = ambient_dimension(rec)
    high_corner1 = svector(n -> n == axis ? place : rec_high_corner[n], N)
    low_corner2 = svector(n -> n == axis ? place : rec_low_corner[n], N)
    rec1 = HyperRectangle(rec_low_corner, high_corner1)
    rec2 = HyperRectangle(low_corner2, rec_high_corner)
    return (rec1, rec2)
end
function Base.split(rec::HyperRectangle, axis)
    place = (high_corner(rec)[axis] + low_corner(rec)[axis]) / 2
    return split(rec, axis, place)
end
function Base.split(rec::HyperRectangle)
    axis = argmax(high_corner(rec) .- low_corner(rec))
    return split(rec, axis)
end

"""
    section(rec::HyperRectangle{D},ax)

Return the dimension `D-1` `HyperRectangle` obtained by deleting the coordinates
at the `ax` dimension.
"""
function section(rec::HyperRectangle{D}, ax::Integer) where {D}
    @assert 1 ≤ ax ≤ D
    return HyperRectangle(deleteat(low_corner(rec), ax), deleteat(high_corner(rec), ax))
end
