"""
    share_interp_data(mode=true)

If set to `true`, neighboring cone interpolants will share their interpolation
data (which are, in this case, computed only once).
"""
function share_interp_data(mode = true)
    @eval _share_interp_data() = $mode
    return mode
end
# FIXME: this option only works in serial due to a "bug" when transfering the
# nodes to the right neighbor in parallel.
_share_interp_data() = false

"""
    use_fftw(mode=true)

Use the [FFTW](http://www.fftw.org) library for computing the Chebyshev transform.
"""
function use_fftw(mode = true)
    @eval _use_fftw() = $mode
    return mode
end
_use_fftw() = true

"""
    use_minimal_conedomain(mode=true)

If set to `true`, the minimal domain for the cone interpolants will be used by
calling [`interpolation_domain`](@ref) during the contruction of the active
cones.

While setting this to `true` reduces the memory footprint by creating
fewer cones, it also increases the time to assemble the `IFGFOperator`.
"""
function use_minimal_conedomain(mode = true)
    mode == false && (@warn "experimental feature: may lead to unexpected behavior")
    @eval _use_minimal_conedomain() = $mode
    return mode
end
_use_minimal_conedomain() = true

# fast invsqrt code taken from here
# https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
@inline function invsqrt(x::Float64)
    y = @fastmath Float64(1 / sqrt(Float32(x)))
    # This is a Newton-Raphson iteration.
    return 1.5y - 0.5x * y * (y * y)
end

"""
    struct ConeDomainSize

Use `ConeDomainSize(k,ds)` to create a function that computes the appropriate
meshsize for a given `ClusterTree` based on the wavenumber `k` and the reference
meshsize `ds`.

Calling `f(node)` will return the meshsize for the given `node`, where `f` is a
`ConeDomainSize` object.
"""
struct ConeDomainSize{N,T,S}
    k::T
    ds::SVector{N,S}
end
ConeDomainSize(k, ds::Tuple) = ConeDomainSize(k, SVector(ds))

"""
    const ACOUSTIC_SIZE_THRESHOLD

Boxes with a size larger than `ACOUSTIC_SIZE_THRESHOLD` times the wavelength are
considered "high-frequency", and their cone-domain size will be refined.
"""
const ACOUSTIC_SIZE_THRESHOLD = Ref(1)

function (f::ConeDomainSize)(node)
    if iszero(f.k)
        return f.ds
    else
        # oscillatory case (e.g. Helmholtz, Maxwell)
        # k: wavenumber
        bbox = container(node)
        w    = maximum(high_corner(bbox) - low_corner(bbox))
        # FIXME: it is not clear when boxes should be considered
        # "high-frequency" so that we need to refine the cone-domain size.
        λ = 2π / f.k
        c = ACOUSTIC_SIZE_THRESHOLD[]
        δ = max(w / (c * λ), 1)
        return f.ds / δ
    end
end

"""
    struct AdmissibilityCondition

Functor used to determine if the interaction between a `target` and a `source`
can be approximated using an interpolation scheme.

A target and source are admissible under this *admissiblility condition* if the
target box lies farther than `r*η` away, where `r` is the radius of the source
box and `η >= 1` is an adjustable parameter. In dimension `N`, the default
choice is `η = √N`.
"""
struct AdmissibilityCondition
    η::Float64
end

function (f::AdmissibilityCondition)(target, source)
    xc   = source |> container |> center
    h    = source |> container |> radius
    bbox = container(target)
    dc   = distance(xc, bbox)
    # if target box is outside a sphere of radius h*η, consider it admissible.
    return dc > f.η * h
end

"""
    _density_type_from_kernel_type(T)

Helper function to compute the expected density `V` type associated with a kernel type
`T`. For `T<:Number`, simply return `T`. For `T<:SMatrix`, return an `SVector`
with elements of the type `eltype(T)` and of length `size(T,2)` so that
multiplying an element of type `T` with an element of type `V` makes sense.
"""
function _density_type_from_kernel_type(T)
    if T <: Number
        return T
    elseif T <: Union{SMatrix,Transpose}
        m, n = size(T)
        return SVector{n,eltype(T)}
    else
        error("kernel type $T not recognized")
    end
end

"""
    wavenumber(K::Function)

For oscillatory kernels, return the characteristic wavenumber (i.e `2π` divided
by he wavelength). For non-oscillatory kernels, return `0`.
"""
function wavenumber end

"""
    increment_index(I::CartesianIndex,k[,n=1])

Increment `I` by `n` along  the dimension `k`. This is equivalent to `I +=
n*eₖ`, where `eₖ` is a vector with with `1` at the  `k`-th coordinate and zeros elsewhere.
"""
function increment_index(I::CartesianIndex, dim::Integer, nb::Integer = 1)
    N = length(I)
    @assert 1 ≤ dim ≤ length(I)
    return I + CartesianIndex(ntuple(i -> i == dim ? nb : 0, N))
end

"""
    decrement_index(I::CartesianIndex,k[,n=1])

Equivalent to [`increment_index`](@ref)(I,k,-n)
"""
function decrement_index(I::CartesianIndex, dim::Integer, nb::Integer = 1)
    N = length(I)
    @assert 1 ≤ dim ≤ length(I)
    return I + CartesianIndex(ntuple(i -> i == dim ? -nb : 0, N))
end

"""
    return_type(f[,args...])

The type returned by `f(args...)`, where `args` is a tuple of types. Falls back
to `Base.promote_op` by default.

A functors of type `T` with a known return type should extend
`return_type(::T,args...)` to avoid relying on `promote_op`.
"""
function return_type(f, args...)
    @debug "using `Base.promote_op` to infer return type. Consider defining `return_type(::typeof($f),args...)`."
    return Base.promote_op(f, args...)
end

"""
    depth(tree,acc=0)

Recursive function to compute the depth of `node` in a a tree-like structure.

Overload this function if your structure has a more efficient way to compute
`depth` (e.g. if it stores it in a field).
"""
function depth(tree, acc = 0)
    if isroot(tree)
        return acc
    else
        depth(parent(tree), acc + 1)
    end
end

"""
    partition_by_depth(tree)

Given a `tree`, return a `partition` vector whose `i`-th entry stores all the nodes in
`tree` with `depth=i-1`. Empty nodes are not added to the partition.
"""
function partition_by_depth(tree)
    T = typeof(tree)
    partition = Vector{Vector{T}}()
    depth = 0
    return _partition_by_depth!(partition, tree, depth)
end

function _partition_by_depth!(partition, tree, depth)
    if length(partition) < depth + 1
        push!(partition, [])
    end
    length(tree) > 0 && push!(partition[depth+1], tree)
    for chd in children(tree)
        _partition_by_depth!(partition, chd, depth + 1)
    end
    return partition
end

"""
    svector(f,n)

Create an `SVector` of length n, computing each element as f(i), where `i ∈
1:n`.
"""
svector(f, n) = ntuple(f, n) |> SVector

# some reasonable defaults for center
center(x::Tuple) = SVector(x)
center(x::SVector) = x
center(x) = coords(x)

"""
    coords(x)

Return an `SVector` with the cartesian coordinates associated to a geometrical
object `x`.
"""
coords(x::Tuple) = SVector(x)
coords(x::SVector) = x

function coords(x::T) where {T}
    if hasfield(T, :coords)
        return getfield(x, :coords)
    else
        error("type $T has no method nor field named `coords`.")
    end
end

"""
    usethreads(mode=true)

Enable/disable thread usage globally for the `IFGF` module.
"""
function usethreads(mode = true)
    @eval _usethreads() = $mode
end
_usethreads() = true

# Inspired by
# https://discourse.julialang.org/t/putting-threads-threads-or-any-macro-in-an-if-statement/41406/7
# Modified so that the macro calls a pure function `usethreads`
macro usethreads(expr::Expr)
    ex = quote
        if IFGF._usethreads()
            Threads.@threads $expr
        else
            $expr
        end
    end
    return esc(ex)
end

macro usespawn(expr::Expr)
    ex = quote
        if IFGF._usethreads()
            Threads.@spawn $expr
        else
            $expr
        end
    end
    return esc(ex)
end

"""
    real_and_imag(v)

Return a view over the real and imaginary parts of a complex vector `v`.

```jldoctest
julia> v = [1.0 + 2.0im, 3.0 + 4.0im];

julia> vr,vi = IFGF.real_and_imag(v)
([1.0, 3.0], [2.0, 4.0])
```
"""
function real_and_imag(v::AbstractMatrix{Complex{T}}) where {T<:Real}
    vfloat = reinterpret(T, v)
    vr = @views vfloat[1:2:end, :]
    vi = @views vfloat[2:2:end, :]
    return vr, vi
end

function real_and_imag(v::AbstractVector{Complex{T}}) where {T<:Real}
    vfloat = reinterpret(T, v)
    vr = @views vfloat[1:2:end]
    vi = @views vfloat[2:2:end]
    return vr, vi
end

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}
const Point2Df = SVector{2,Float32}
const Point3Df = SVector{3,Float32}

function simd_split(I::UnitRange, sz)
    i1, il = first(I), last(I)
    n = div(length(I), sz)
    return range(i1; length = n, step = sz), (i1+n*sz):il
end

"""
    float_type(x)

Floating type used to represent the underlying structure (e.g. `Float32`,
`Float64`).
"""
float_type(::Type{Complex{T}}) where {T} = T
float_type(::Type{SVector{<:,T}}) where {T<:Real} = T

"""
    _unsafe_wrap_vector_of_sarray(A::Array)

Wrap the memory in `A` so that its first `n-1` dimensions are interpreted as an
`SArray`, returning a `Vector{<:SArray}` type.

This function uses `unsafe_wrap`, and therefore is unsafe for the same reasons
`unsafe_wrap` is.
"""
function _unsafe_wrap_vector_of_sarray(A::Array{T,N}) where {T,N}
    @assert isbitstype(T) "type $T is not a bitstype"
    dims = ntuple(i -> size(A, i), N - 1)
    n    = size(A, N) # size of last dimension
    V    = SArray{Tuple{dims...},T,N - 1,prod(dims)}
    return unsafe_wrap(Array, convert(Ptr{V}, pointer(A)), n)
end

function points_on_unit_sphere(N, T = Float64)
    pts = Matrix{T}(undef, 3, N)
    phi = π * (3 - sqrt(5)) # golden angle in radians
    for i in 1:N
        y = 1 - ((i - 1) / (N - 1)) * 2
        radius = sqrt(1 - y^2)
        theta = phi * i
        pts[1, i] = cos(theta) * radius
        pts[2, i] = y
        pts[3, i] = sin(theta) * radius
    end
    return pts
end

function points_on_circle(n)
    pts = Vector{Point2D}(undef, n)
    h = 2 * π / n
    for i in 1:n
        angle = i * h
        x = cos(angle)
        y = sin(angle)
        pts[i] = SVector(x, y)
    end
    return pts
end
