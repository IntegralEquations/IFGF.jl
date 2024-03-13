@doc raw"""
    laplace3d(targets, sources; charges = nothing, dipvecs = nothing, grad =
    false, kwargs...)

Compute the sum

```math
u(\boldsymbol{x}_i) = \sum_{j=1}^n \frac{c_j}{|| \boldsymbol{x}_i - \boldsymbol{y}_j ||} -
\frac{(\boldsymbol{x}_i - \boldsymbol{y}_j)^\top v_j}{|| \boldsymbol{x}_i - \boldsymbol{y}_j ||^3} ,
```

where $c_j$ are the `charges` and `v_j` are the `dipvecs`.

## Input:

- `targets::Matrix{T}`: `3 x n` matrix of target points.
- `sources::Matrix{T}`: `3 x m` matrix of source points.
- `charges::Vector{T}`: vector of `n` charges.
- `dipvecs::Matrix{T}`: `3 x m` matrix of dipole vectors.
- `grad::Bool`: if `true`, the gradient is computed instead
- `kwargs...`: additional keyword arguments passed to [`assemble_ifgf`](@ref)
  (e.g. `tol`)

## Output

- `u`: the potential at the target points if `grad=false`, or the gradient as a
  `3 x n` matrix if `grad=true`.

"""
function laplace3d(
    targets::Matrix{T},
    sources::Matrix{T};
    charges = nothing,
    dipvecs = nothing,
    grad = false,
    kwargs...,
) where {T<:Real}
    L = plan_laplace3d(targets, sources; charges, dipvecs, grad, kwargs...)
    return laplace3d(L; charges, dipvecs, grad)
end

"""
    plan_laplace3d(targets, sources; charges = nothing, dipvecs = nothing, grad =
    false, kwargs...)

Similar to [`laplace3d`](@ref), but returns a plan in the form of an `IFGFOp``
object instead of the result. Calling `laplace3d` with `plan` as the first
argument will then compute the result.
"""
function plan_laplace3d(
    targets::Matrix{T},
    sources::Matrix{T};
    charges = nothing,
    dipvecs = nothing,
    grad = false,
    kwargs...,
) where {T<:Real}
    pde = Laplace(; dim = 3)
    # check arguments for consistency
    isnothing(charges) &&
        isnothing(dipvecs) &&
        error("either charges or dipvecs should be provided")
    @assert size(targets, 1) == 3 "targets must be a 3 x n matrix"
    @assert size(sources, 1) == 3 "sources must be a 3 x m matrix"
    L = _plan_forward_map(pde, targets, sources, charges, dipvecs, grad; kwargs...)
    return L
end

"""
    laplace3d!(out, L::IFGFOp; charges = nothing, dipvecs = nothing, grad =
    false)

In-place version of [`laplace3d`](@ref).
"""
function laplace3d!(out, L::IFGFOp; charges = nothing, dipvecs = nothing, grad = false)
    # check that arguments adhere to the API
    @assert L.kernel.pde isa Laplace{3} "Laplace kernel required"
    # charges
    if !isnothing(charges)
        @assert isa(charges, Vector) "charges must be a vector"
        @assert length(charges) == size(L, 2) "charges must have length equal to the number of targets"
    end
    # dipvecs
    dipvecs_ = if !isnothing(dipvecs)
        @assert isa(dipvecs, Matrix) && size(dipvecs, 1) == 3 "dipvecs must be a 3 x m matrix"
        @assert size(dipvecs, 2) == size(L, 2) "dipvecs must have the same number of columns as the number of targets"
        _unsafe_wrap_vector_of_sarray(dipvecs)
    else
        dipvecs
    end
    # output
    out_ = if grad
        @assert isa(out, Matrix) && size(out, 1) == 3 && size(out, 2) == size(L, 1) "out must be a 3 x n matrix"
        _unsafe_wrap_vector_of_sarray(out)
    else
        @assert isa(out, Vector) && length(out) == size(L, 1) "out must be a vector of length equal to the number of targets"
        out
    end
    _forward_map!(out_, L, charges, dipvecs_, grad, "charges", "dipvecs")
    return out
end

function laplace3d(L::IFGFOp; charges = nothing, dipvecs = nothing, grad = false)
    isnothing(charges) &&
        isnothing(dipvecs) &&
        error("either charges or dipvecs should be provided")
    m, n = size(L)
    T = isnothing(charges) ? eltype(dipvecs) : eltype(charges)
    out = grad ? zeros(T, 3, m) : zeros(T, n)
    return laplace3d!(out, L; charges, dipvecs, grad)
end

"""
    struct Laplace{N}

Laplace equation in `N` dimension: Δu = 0.
"""
struct Laplace{N} <: AbstractPDE{N} end

function Laplace(; dim)
    dim == 2 && (error("2D Laplace not support"))
    return Laplace{dim}()
end

default_density_eltype(::Laplace) = Float64

function centered_factor(::PDEKernel{Laplace{2}}, x, Y)
    return 1
end

function centered_factor(::PDEKernel{Laplace{3}}, x, Y)
    yc = center(Y)
    d  = norm(x - yc)
    return 1 / d
end

wavenumber(::PDEKernel{<:Laplace}) = 0

function (SL::SingleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (log(d))
    elseif N == 3
        return filter * (1 / d)
    else
        error("Not implemented for N = $N")
    end
end

function (::GradSingleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (1 / (d^2) * r)
    elseif N == 3
        return filter * (-1 / (d^3) * r)
    else
        error("Not implemented for N = $N")
    end
end

function (::DoubleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 / (d^2) * transpose(r))
    elseif N == 3
        return filter * (1 / (d^3) * transpose(r))
    else
        error("Not implemented for N = $N")
    end
end

function (::CombinedFieldKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * SVector{N + 1,T}(log(d), -r[1] / d^2, -r[2] / d^2) |> transpose
    elseif N == 3
        return filter * SVector{N + 1,T}(1 / d, r[1] / d^3, r[2] / d^3, r[3] / d^3) |>
               transpose
    else
        error("Not implemented for N = $N")
    end
end

function (::HessianSingleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 / (d^2) * ((I - 2 * r * transpose(r) / d^2)))
    elseif N == 3
        RRT = r * transpose(r) # r ⊗ rᵗ
        return filter * (1 / (d^3) * ((I - 3 * RRT / d^2)))
    end
end

function near_interaction!(
    C::AbstractVector{T},
    ::SingleLayerKernel{Laplace{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{T},
    I::UnitRange,
    J::UnitRange,
) where {T}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    @views laplace3d_sl_simd!(C[I], Xm[:, I], Ym[:, J], σ[J])
end

function laplace3d_sl_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1, i] - Y[1, j])^2
            d2 += (X[2, i] - Y[2, j])^2
            d2 += (X[3, i] - Y[3, j])^2
            C[i] += (!iszero(d2)) * (inv(sqrt(d2)) * σ[j])
        end
    end
    return C
end

function near_interaction!(
    C::AbstractVector{T},
    ::DoubleLayerKernel{Laplace{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{SVector{3,T}},
    I::UnitRange,
    J::UnitRange,
) where {T}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    σm = reinterpret(reshape, T, σ)
    @views laplace3d_dl_simd!(C[I], Xm[:, I], Ym[:, J], σm[:, J])
end

function laplace3d_dl_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1, r2, r3 = X[1, i] - Y[1, j], X[2, i] - Y[2, j], X[3, i] - Y[3, j]
            r_dot_σ = r1 * σ[1, j] + r2 * σ[2, j] + r3 * σ[3, j]
            d2 = r1^2 + r2^2 + r3^2
            d = sqrt(d2)
            C[i] += (!iszero(d2)) * (inv(d^3) * r_dot_σ)
        end
    end
    return C
end

function near_interaction!(
    C::AbstractVector{SVector{3,T}},
    ::GradSingleLayerKernel{Laplace{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{T},
    I::UnitRange,
    J::UnitRange,
) where {T}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    Cm = reinterpret(reshape, T, C)
    @views laplace3d_gradsl_simd!(Cm[:, I], Xm[:, I], Ym[:, J], σ[J])
end

function laplace3d_gradsl_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1, r2, r3 = X[1, i] - Y[1, j], X[2, i] - Y[2, j], X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            d = sqrt(d2)
            v = (!iszero(d2)) * (inv(d^3)) * σ[j]
            C[1, i] -= v * r1
            C[2, i] -= v * r2
            C[3, i] -= v * r3
        end
    end
    return C
end

function near_interaction!(
    C::AbstractVector{SVector{3,T}},
    ::HessianSingleLayerKernel{Laplace{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{SVector{3,T}},
    I::UnitRange,
    J::UnitRange,
) where {T}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    σm = reinterpret(reshape, T, σ)
    Cm = reinterpret(reshape, T, C)
    @views laplace3d_hessiansl_simd!(Cm[:, I], Xm[:, I], Ym[:, J], σm[:, J])
end

function laplace3d_hessiansl_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1, r2, r3 = X[1, i] - Y[1, j], X[2, i] - Y[2, j], X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            d = sqrt(d2)
            r_dot_σ = r1 * σ[1, j] + r2 * σ[2, j] + r3 * σ[3, j]
            C[1, i] += (!iszero(d2)) * (σ[1, j] / d^3 - 3 * r1 * r_dot_σ / d^5)
            C[2, i] += (!iszero(d2)) * (σ[2, j] / d^3 - 3 * r2 * r_dot_σ / d^5)
            C[3, i] += (!iszero(d2)) * (σ[3, j] / d^3 - 3 * r3 * r_dot_σ / d^5)
        end
    end
    return C
end

function near_interaction!(
    C::AbstractVector{T},
    ::CombinedFieldKernel{Laplace{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{SVector{4,T}},
    I::UnitRange,
    J::UnitRange,
) where {T}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    σm = reinterpret(reshape, T, σ)
    @views laplace3d_combinedfield_simd!(C[I], Xm[:, I], Ym[:, J], σm[:, J])
end

function laplace3d_combinedfield_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1, r2, r3 = X[1, i] - Y[1, j], X[2, i] - Y[2, j], X[3, i] - Y[3, j]
            r_dot_σ = r1 * σ[2, j] + r2 * σ[3, j] + r3 * σ[4, j]
            d2 = r1^2 + r2^2 + r3^2
            d = sqrt(d2)
            C[i] += (!iszero(d2)) * (inv(d) * σ[1, j] + inv(d^3) * r_dot_σ)
        end
    end
    return C
end
