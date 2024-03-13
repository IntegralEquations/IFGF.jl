@doc raw"""
    stokes3d(targets, sources; stoklet = nothing, strslet = nothing, strsvec =
    nothing, grad = false, kwargs...)

Compute the sum

```math
u(\boldsymbol{x}_i) = \sum_{j=1}^n \left( \frac{I}{d} + \frac{\boldsymbol{r}_{ij}\boldsymbol{r}_{ij}^\top}{2d_{ij}^3} \right) \boldsymbol{\sigma}_j +
\frac{3(\boldsymbol{\nu}_j \cdot \boldsymbol{r}_{ij})(\boldsymbol{\mu}_j\cdot\boldsymbol{r}_{ij}) \boldsymbol{r}_{ij}}{d_{ij}^5},
```

where $\boldsymbol{r}_{ij} = |\boldsymbol{x}_i - \boldsymbol{y}_j|$, $d_{ij} =
|\boldsymbol{r_{ij}}|$, $\boldsymbol{\sigma}_j$ are the Stokeslet strengths,
$\boldsymbol{\mu}_j$ are the stresslet strengths, and $\boldsymbol{\nu}_j$ are the
stresslet orientations.

## Input:

- `targets::Matrix{T}`: `3 x n` matrix of target points.
- `sources::Matrix{T}`: `3 x m` matrix of source points.
- `stoklet::Matrix{T}`: `3 × n` matrix of stokeslets
- `strslet::Matrix{T}`: `6 x m` matrix of stresslet densities (indices `1:3`)
  and orientations (indices `4:6`)
- `grad::Bool`: if `true`, the gradient is computed instead
- `kwargs...`: additional keyword arguments passed to [`assemble_ifgf`](@ref).

## Output

- `u`: `3 x n` matrix giving the velocity at the target points if `grad=false`,
  or `3 x 3 × n` matrix representing the gradient of `u` if `grad=true`.

"""
function stokes3d(
    targets::Matrix{T},
    sources::Matrix{T};
    stoklet = nothing,
    strslet = nothing,
    grad = false,
    kwargs...,
) where {T<:Real}
    L = plan_stokes3d(targets, sources; stoklet, strslet, grad, kwargs...)
    return stokes3d(L; stoklet, strslet, grad)
end

function plan_stokes3d(
    targets::Matrix{T},
    sources::Matrix{T};
    stoklet = nothing,
    strslet = nothing,
    grad = false,
    kwargs...,
) where {T<:Real}
    pde = Stokes(; dim = 3)
    # check arguments for consistency
    isnothing(stoklet) &&
        isnothing(strslet) &&
        error("either stoklet or strslet should be provided")
    @assert size(targets, 1) == 3 "targets must be a 3 x n matrix"
    @assert size(sources, 1) == 3 "sources must be a 3 x m matrix"
    grad && error("not (yet) supported")
    L = _plan_forward_map(pde, targets, sources, stoklet, strslet, grad; kwargs...)
    return L
end

function stokes3d!(out, L::IFGFOp; stoklet = nothing, strslet = nothing, grad = false)
    # check that arguments adhere to the API
    @assert L.kernel.pde isa Stokes{3} "Stokes kernel required"
    # stoklet
    stoklet_ = if !isnothing(stoklet)
        @assert isa(stoklet, Matrix) && size(stoklet, 1) == 3 "stoklet must be a 3 x n matrix"
        @assert size(stoklet, 2) == size(L, 2) "stoklet must have as many columns as the number of targets"
        _unsafe_wrap_vector_of_sarray(stoklet)
    else
        stoklet
    end
    # strslet
    strslet_ = if !isnothing(strslet)
        @assert isa(strslet, Matrix) && size(strslet, 1) == 6 "strslet must be a 6 x m matrix"
        @assert size(strslet, 2) == size(L, 2) "strslet/strsvec must have as many columns as the number of targets"
        _unsafe_wrap_vector_of_sarray(strslet)
    else
        strslet
    end
    # output
    out_ = if grad
        error("not (yet) supported")
    else
        @assert isa(out, Matrix) && size(out, 1) == 3 && size(out, 2) == size(L, 1) "out must be a 3 x n matrix"
        _unsafe_wrap_vector_of_sarray(out)
    end
    _forward_map!(out_, L, stoklet_, strslet_, grad, "stoklet", "strslet")
    return out
end

function stokes3d(L::IFGFOp; stoklet = nothing, strslet = nothing, grad = false)
    isnothing(stoklet) &&
        isnothing(strslet) &&
        error("either stoklet or strslet should be provided")
    m, n = size(L)
    T = isnothing(stoklet) ? eltype(strslet) : eltype(stoklet)
    out = grad ? zeros(T, 3, 3, m) : zeros(T, 3, n)
    return stokes3d!(out, L; stoklet, strslet, grad)
end

"""
    struct Stokes{N}

Stokes equation in `N` dimension.
"""
struct Stokes{N} <: AbstractPDE{N} end

Stokes(; dim = 3) = Stokes{dim}()

default_density_eltype(::Stokes{N}) where {N} = SVector{N,Float64}

IFGF.wavenumber(::PDEKernel{<:Stokes}) = 0

function centered_factor(::PDEKernel{Stokes{3}}, x, Y)
    yc = center(Y)
    d  = norm(x - yc)
    return 1 / d
end

function (SL::SingleLayerKernel{<:Stokes{N}})(target, source, r = target - source) where {N}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        γ = -log(d)
    elseif N == 3
        γ = 1 / d
    end
    return filter * ((γ * I + r * transpose(r) / d^N) / (N - 1))
end

function near_interaction!(
    C::AbstractVector{T},
    ::SingleLayerKernel{<:Stokes{3}},
    Xp::AbstractVector{T},
    Yp::Vector{T},
    σ::Vector{T},
    I::UnitRange,
    J::UnitRange,
) where {T<:SVector{3,F}} where {F<:Real}
    Xm = reinterpret(reshape, F, Xp)
    Ym = reinterpret(reshape, F, Yp)
    Cm = reinterpret(reshape, F, C)
    σm = reinterpret(reshape, F, σ)
    @views stokes3d_sl_simd!(Cm[:, I], Xm[:, I], Ym[:, J], σm[:, J])
    return C
end

function stokes3d_sl_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1 = X[1, i] - Y[1, j]
            r2 = X[2, i] - Y[2, j]
            r3 = X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            filter = !iszero(d2)
            invd = inv(sqrt(d2))
            γ = (r1 * σ[1, j] + r2 * σ[2, j] + r3 * σ[3, j]) * invd^3
            C[1, i] += filter * ((σ[1, j] * invd + r1 * γ) / 2)
            C[2, i] += filter * ((σ[2, j] * invd + r2 * γ) / 2)
            C[3, i] += filter * ((σ[3, j] * invd + r3 * γ) / 2)
        end
    end
    return C
end

# Double Layer Kernel
struct Stresslet{N,T} <: AbstractArray{T,3}
    r::SVector{N,T}
end

Base.size(::Stresslet{N}) where {N} = (N, N, N)
Base.eltype(::Stresslet{<:,T}) where {T} = T

function Base.getindex(S::Stresslet{3}, i, j, k)
    r = S.r
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    return filter * (3 * r[i] * r[j] * r[k] / d^5)
end

function Base.:*(S::Stresslet{3}, x::SVector{6})
    μ = x[1:3]
    ν = x[4:6]
    r = S.r
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    return filter * (3 * dot(μ, S.r) * dot(ν, S.r) / d^5) * r
end
Base.:*(S::Stresslet{3}, x::AbstractVector) = S * SVector{6}(x)

function (DL::DoubleLayerKernel{<:Stokes{3}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    return Stresslet(target - source)
end

function near_interaction!(
    C::AbstractVector{SVector{3,F}},
    ::DoubleLayerKernel{<:Stokes{3}},
    Xp::AbstractVector{SVector{3,F}},
    Yp::Vector{SVector{3,F}},
    σ::Vector{SVector{6,F}},
    I::UnitRange,
    J::UnitRange,
) where {F<:Real}
    Xm = reinterpret(reshape, F, Xp)
    Ym = reinterpret(reshape, F, Yp)
    Cm = reinterpret(reshape, F, C)
    σm = reinterpret(reshape, F, σ)
    @views stokes3d_dl_simd!(Cm[:, I], Xm[:, I], Ym[:, J], σm[:, J])
    return C
end

function stokes3d_dl_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1 = X[1, i] - Y[1, j]
            r2 = X[2, i] - Y[2, j]
            r3 = X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            filter = !iszero(d2)
            invd = inv(sqrt(d2))
            μdotn = (r1 * σ[1, j] + r2 * σ[2, j] + r3 * σ[3, j])
            νdotn = (r1 * σ[4, j] + r2 * σ[5, j] + r3 * σ[6, j])
            γ = 3 * (μdotn * νdotn) * invd^5
            C[1, i] += filter * (r1 * γ)
            C[2, i] += filter * (r2 * γ)
            C[3, i] += filter * (r3 * γ)
        end
    end
    return C
end

function (::CombinedFieldKernel{Stokes{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    if N == 3
        return CombinedFieldStokes(r)
    else
        error("Not implemented for N = $N")
    end
end

struct CombinedFieldStokes{N,T} <: AbstractArray{T,3}
    r::SVector{N,T}
end

Base.size(::CombinedFieldStokes{3}) = (3)
Base.eltype(::CombinedFieldStokes{<:,T}) where {T} = T

function Base.getindex(S::CombinedFieldStokes{3}, i, j, k)
    r = S.r
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    return filter * (3 * r[i] * r[j] * r[k] / d^5)
end

function Base.:*(S::CombinedFieldStokes{3}, x::SVector{9})
    σ = x[1:3]
    μ = x[4:6]
    ν = x[7:9]
    r = S.r
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    G = (1 / d * σ + r * transpose(r) * σ / d^3) / 2
    dG = (3 * dot(μ, S.r) * dot(ν, S.r) / d^5) * r
    return filter * (G + dG)
end
Base.:*(S::CombinedFieldStokes{3}, x::AbstractVector) = S * SVector{9}(x)

function near_interaction!(
    C::AbstractVector{SVector{3,F}},
    ::CombinedFieldKernel{<:Stokes{3}},
    Xp::AbstractVector{SVector{3,F}},
    Yp::Vector{SVector{3,F}},
    σ::Vector{SVector{9,F}},
    I::UnitRange,
    J::UnitRange,
) where {F<:Real}
    Xm = reinterpret(reshape, F, Xp)
    Ym = reinterpret(reshape, F, Yp)
    Cm = reinterpret(reshape, F, C)
    σm = reinterpret(reshape, F, σ)
    @views stokes3d_cf_simd!(Cm[:, I], Xm[:, I], Ym[:, J], σm[:, J])
end

function stokes3d_cf_simd!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            r1 = X[1, i] - Y[1, j]
            r2 = X[2, i] - Y[2, j]
            r3 = X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            filter = !iszero(d2)
            invd = inv(sqrt(d2))
            γ = (r1 * σ[1, j] + r2 * σ[2, j] + r3 * σ[3, j]) * invd^3 / 2
            μdotn = (r1 * σ[4, j] + r2 * σ[5, j] + r3 * σ[6, j])
            νdotn = (r1 * σ[7, j] + r2 * σ[8, j] + r3 * σ[9, j])
            γ += 3 * (μdotn * νdotn) * invd^5
            C[1, i] += filter * (σ[1, j] * invd / 2 + r1 * γ)
            C[2, i] += filter * (σ[2, j] * invd / 2 + r2 * γ)
            C[3, i] += filter * (σ[3, j] * invd / 2 + r3 * γ)
        end
    end
    return C
end

# TODO: implement grad for stokes
