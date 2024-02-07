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
    use_vectorized_chebeval(mode=true)

Use a vectorized version of the Chebyshev evaluation routine. Calling this
function will redefine `chebeval` to point to either `chebeval_vec` or
`chebeval_novec`, and thus may trigger code recompilation.
"""
function use_vectorized_chebeval(mode = true)
    if mode
        @eval chebeval(args...) = chebeval_vec(args...)
    else
        @eval chebeval(args...) = chebeval_novec(args...)
    end
    return nothing
end

"""
    chebeval(coefs,x,rec[,sz])

Evaluate the Chebyshev polynomial defined on `rec` with coefficients given by
`coefs` at the point `x`. If the size of `coefs` is known statically, its size
can be passed as a `Val` using the `sz` argument.
"""
chebeval(args...) = chebeval_novec(args...)

"""
    use_minimal_conedomain(mode=true)

If set to `true`, the minimal domain for the cone interpolants will be used by
calling [`interpolation_domain`](@ref) during the contruction of the active
cones.

While setting this to `true` reduces the memory footprint by creating
fewer cones, it also increases the time to assemble the `IFGFOperator`.
"""
function use_minimal_conedomain(mode = true)
    @eval _use_minimal_conedomain() = $mode
    return mode
end
_use_minimal_conedomain() = false

# fast invsqrt code taken from here
# https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
@inline function invsqrt(x::Float64)
    y = @fastmath Float64(1 / sqrt(Float32(x)))
    # This is a Newton-Raphson iteration.
    return 1.5y - 0.5x * y * (y * y)
end

"""
    cone_domain_size_func(Δs₀::NTuple{N,T},k)

Returns an anonymous function `(node) -> Δs` that computes an appropriate size
`Δs` for the interpolation domain of `node` given an intial size `Δs₀` and a
wavenumber `k`. The function is constructed so as to scale `Δs₀` by the inverse
of the acoustic size of `node`.
"""
function cone_domain_size_func(k, ds)
    if k == 0
        func = (node) -> ds
    else
        # oscillatory case (e.g. Helmholtz, Maxwell)
        # k: wavenumber
        func = (node) -> begin
            bbox = IFGF.container(node)
            w    = maximum(high_corner(bbox) - low_corner(bbox))
            δ    = max(k * w / 2, 1)
            ds ./ δ
        end
    end
    return func
end

function cone_domain_size_func(k, ds::Number)
    if k == 0
        func = (node) -> begin
            N = ambient_dimension(node)
            ntuple(i -> ds, N)
        end
    else
        # oscillatory case (e.g. Helmholtz, Maxwell)
        # k: wavenumber
        func = (node) -> begin
            N = ambient_dimension(node)
            ds_tup = ntuple(i -> ds, N)
            bbox = IFGF.container(node)
            w = maximum(high_corner(bbox) - low_corner(bbox))
            δ = max(k * w / 2, 1)
            ds_tup ./ δ
        end
    end
    return func
end

"""
    modified_admissible_condition(target,source,[η])

A target and source are admissible under the *modified admissiblility condition*
(MAC) if the target box lies farther than `r*η` away, where `r` is the radius of
the source box and `η >= 1` is an adjustable parameter. By default, `η = N /
√N`, where `N` is the ambient dimension.
"""
function modified_admissibility_condition(target, source, η)
    # compute distance between source center and target box
    xc   = source |> container |> center
    h    = source |> container |> radius
    bbox = container(target)
    dc   = distance(xc, bbox)
    # if target box is outside a sphere of radius h*η, consider it admissible.
    return dc > η * h
end

function modified_admissibility_condition(target, source)
    N = ambient_dimension(target)
    η = N / sqrt(N)
    return modified_admissibility_condition(target, source, η)
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
    elseif T <: Union{SMatrix,Adjoint}
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
