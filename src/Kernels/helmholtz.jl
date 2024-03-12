@doc raw"""
    helmholtz3d(k, targets, sources; charges = nothing, dipvecs = nothing, grad =
    false, kwargs...)

Compute the sum

```math
u(\boldsymbol{x}_i) = \sum_{j=1}^n \frac{c_j e^{i k || \boldsymbol{x}_i - \boldsymbol{y}_j ||}}{|| \boldsymbol{x}_i - \boldsymbol{y}_j ||} + \nabla_{\boldsymbol{y}}^\top \left( \frac{e^{i k || \boldsymbol{x}_i - \boldsymbol{y}_j ||}}{|| \boldsymbol{x}_i - \boldsymbol{y}_j ||} \right) v_j,
```

where $c_j$ are the `charges` and `v_j` are the `dipvecs`.

## Input:

- `k`: the wavenumber.
- `targets::Matrix{T}`: `3 x n` matrix of target points.
- `sources::Matrix{T}`: `3 x m` matrix of source points.
- `charges::Vector{Complex{T}}`: vector of `n` charges.
- `dipvecs::Matrix{Complex{T}}`: `3 x m` matrix of dipole vectors.
- `grad::Bool`: if `true`, the gradient is computed instead
- `kwargs...`: additional keyword arguments passed to [`assemble_ifgf`](@ref).

## Output

- `u`: the potential at the target points if `grad=false`, or the gradient as a
  `3 x n` matrix if `grad=true`.

"""
function helmholtz3d(
    k::Number,
    targets::Matrix{T},
    sources::Matrix{T};
    charges = nothing,
    dipvecs = nothing,
    grad = false,
    kwargs...,
) where {T<:Real}
    L = plan_helmholtz3d(k, targets, sources; charges, dipvecs, grad, kwargs...)
    return helmholtz3d(L; charges, dipvecs, grad)
end

function plan_helmholtz3d(
    k::Number,
    targets::Matrix{T},
    sources::Matrix{T};
    charges = nothing,
    dipvecs = nothing,
    grad = false,
    kwargs...,
) where {T<:Real}
    pde = Helmholtz(; dim = 3, k)
    # check arguments for consistency and convert to appropriate types
    isnothing(charges) &&
        isnothing(dipvecs) &&
        error("either charges or dipvecs should be provided")
    @assert size(targets, 1) == 3 "targets must be a 3 x n matrix"
    @assert size(sources, 1) == 3 "sources must be a 3 x m matrix"
    V = SVector{3,T}
    targets_ = unsafe_wrap(Array, convert(Ptr{V}, pointer(targets)), size(targets, 2))
    sources_ = unsafe_wrap(Array, convert(Ptr{V}, pointer(sources)), size(sources, 2))
    L = _plan_forward_map(pde, targets_, sources_, charges, dipvecs, grad; kwargs...)
    return L
end

function helmholtz3d!(out, L::IFGFOp; charges = nothing, dipvecs = nothing, grad = false)
    # check that arguments adhere to the API
    @assert L.kernel.pde isa Helmholtz{3} "Helmholtz kernel required"
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

function helmholtz3d(L::IFGFOp; charges = nothing, dipvecs = nothing, grad = false)
    isnothing(charges) &&
        isnothing(dipvecs) &&
        error("either charges or dipvecs should be provided")
    m, n = size(L)
    T = isnothing(charges) ? eltype(dipvecs) : eltype(charges)
    out = grad ? zeros(T, 3, m) : zeros(T, n)
    return helmholtz3d!(out, L; charges, dipvecs, grad)
end

"""
    struct Helmholtz{N, T}

Helmholtz equation in `N` dimension: Δu + k²u= 0. `k` is the wavenumber of
numeric type T.
"""
struct Helmholtz{N,T} <: AbstractPDE{N}
    k::T
end

function Base.show(io::IO, pde::Helmholtz{N}) where {N}
    return print(io, "Helmholtz $(N)d")
end

Helmholtz(; dim, k::T) where {T} = Helmholtz{dim,T}(k)

default_density_eltype(::Helmholtz) = ComplexF64

function centered_factor(K::PDEKernel{Helmholtz{2,T}}, x, Y) where {T}
    yc = center(Y)
    d  = norm(x - yc)
    # return exp(im*K.pde.k*d)/sqrt(d)
    return hankelh1(0, K.pde.k * d)
end

function centered_factor(K::PDEKernel{Helmholtz{3,T}}, x, Y) where {T}
    yc = center(Y)
    d  = norm(x - yc)
    return exp(im * K.pde.k * d) / d
end

function IFGF.transfer_factor(K::PDEKernel{Helmholtz{3,T}}, x, Y) where {T}
    yc = IFGF.center(Y)
    yp = IFGF.center(IFGF.parent(Y))
    d  = norm(x - yc)
    dp = norm(x - yp)
    return exp(im * K.pde.k * (d - dp)) * dp / d
end

wavenumber(K::PDEKernel{<:Helmholtz}) = K.pde.k

function (K::SingleLayerKernel{<:Helmholtz{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (hankelh1(0, K.pde.k * d))
    elseif N == 3
        return filter * (exp(im * K.pde.k * d) / d)
    else
        error("Not implemented for N = $N")
    end
end

function (K::GradSingleLayerKernel{<:Helmholtz{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = K.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 * k / d * hankelh1(1, k * d) * r)
    elseif N == 3
        return filter * (-1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * r)
    else
        error("Not implemented for N = $N")
    end
end

function (K::DoubleLayerKernel{<:Helmholtz{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = K.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (1 * k / d * hankelh1(1, k * d) * transpose(r))
    elseif N == 3
        return filter * (1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * transpose(r))
    else
        error("Not implemented for N = $N")
    end
end

function (HS::HessianSingleLayerKernel{<:Helmholtz{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = HS.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        RRT = r * transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        val = (-1 * k^2 / d^2 * hankelh1(2, k * d) * RRT + k / d * hankelh1(1, k * d) * I)
        return filter * val
    elseif N == 3
        RRT = r * transpose(r) # r ⊗ rᵗ
        term1 = 1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * I
        term2 = RRT / d * exp(im * k * d) / d^4 * (3 * (d * im * k - 1) + d^2 * k^2)
        return filter * (term1 + term2)
    end
end

function (CF::CombinedFieldKernel{<:Helmholtz{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = CF.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        dbllayer = (1 * k / d * hankelh1(1, k * d) * r)
        return filter *
               SVector{N + 1,Complex{T}}(hankelh1(0, k * d), dbllayer[1], dbllayer[2]) |>
               transpose
    elseif N == 3
        dbllayer = (1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * r)
        return filter * SVector{N + 1,Complex{T}}(
            exp(im * k * d) / d,
            dbllayer[1],
            dbllayer[2],
            dbllayer[3],
        ) |> transpose
    else
        error("Not implemented for N = $N")
    end
end

function near_interaction!(
    C::AbstractVector{Complex{T}},
    K::SingleLayerKernel{<:Helmholtz{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{Complex{T}},
    I::UnitRange,
    J::UnitRange,
) where {T<:Real}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    k = K.pde.k
    @views helmholtz3d_sl_simd!(C[I], Xm[:, I], Ym[:, J], σ[J], k)
    return C
end

function helmholtz3d_sl_simd!(C, X, Y, σ, k)
    m, n = size(X, 2), size(Y, 2)
    C_r, C_i = real_and_imag(C)
    σ_r, σ_i = real_and_imag(σ)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1, i] - Y[1, j])^2
            d2 += (X[2, i] - Y[2, j])^2
            d2 += (X[3, i] - Y[3, j])^2
            d = sqrt(d2)
            s, c = sincos(k * d)
            zr = inv(d) * c
            zi = inv(d) * s
            C_r[i] += (!iszero(d)) * (zr * σ_r[j] - zi * σ_i[j])
            C_i[i] += (!iszero(d)) * (zi * σ_r[j] + zr * σ_i[j])
        end
    end
    return C
end

function near_interaction!(
    C::AbstractVector{Complex{T}},
    K::DoubleLayerKernel{<:Helmholtz{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{SVector{3,Complex{T}}},
    I::UnitRange,
    J::UnitRange,
) where {T<:Real}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    σm = reinterpret(reshape, Complex{T}, σ)
    k = K.pde.k
    @views helmholtz3d_dl_simd!(C[I], Xm[:, I], Ym[:, J], σm[:, J], k)
    return C
end

function helmholtz3d_dl_simd!(C, X, Y, σ, k)
    m, n = size(X, 2), size(Y, 2)
    C_r, C_i = real_and_imag(C)
    σ_r, σ_i = real_and_imag(σ)
    @turbo for j in 1:n
        for i in 1:m
            r1, r2, r3 = X[1, i] - Y[1, j], X[2, i] - Y[2, j], X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            d = sqrt(d2)
            filter = !(d ≤ SAME_POINT_TOLERANCE)
            s, c = sincos(k * d)
            zr = (inv(d) * c + k * s) / d^2
            zi = (inv(d) * s - k * c) / d^2
            r_dot_σr = r1 * σ_r[1, j] + r2 * σ_r[2, j] + r3 * σ_r[3, j]
            r_dot_σi = r1 * σ_i[1, j] + r2 * σ_i[2, j] + r3 * σ_i[3, j]
            C_r[i] += filter * (zr * r_dot_σr - zi * r_dot_σi)
            C_i[i] += filter * (zi * r_dot_σr + zr * r_dot_σi)
        end
    end
    return C
end

function near_interaction!(
    C::AbstractVector{Complex{T}},
    K::CombinedFieldKernel{<:Helmholtz{3}},
    Xp::AbstractVector{SVector{3,T}},
    Yp::Vector{SVector{3,T}},
    σ::Vector{SVector{4,Complex{T}}},
    I::UnitRange,
    J::UnitRange,
) where {T}
    Xm = reinterpret(reshape, T, Xp)
    Ym = reinterpret(reshape, T, Yp)
    σm = reinterpret(reshape, Complex{T}, σ)
    k = K.pde.k
    @views helmholtz3d_combinedfield_simd!(C[I], Xm[:, I], Ym[:, J], σm[:, J], k)
end

function helmholtz3d_combinedfield_simd!(C, X, Y, σ, k)
    m, n = size(X, 2), size(Y, 2)
    C_r, C_i = real_and_imag(C)
    σ_r, σ_i = real_and_imag(σ)
    @turbo for j in 1:n
        for i in 1:m
            r1, r2, r3 = X[1, i] - Y[1, j], X[2, i] - Y[2, j], X[3, i] - Y[3, j]
            d2 = r1^2 + r2^2 + r3^2
            d = sqrt(d2)
            filter = !(d ≤ SAME_POINT_TOLERANCE)
            s, c = sincos(k * d)
            zr = (inv(d) * c + k * s) / d^2
            zi = (inv(d) * s - k * c) / d^2
            r_dot_σr = r1 * σ_r[2, j] + r2 * σ_r[3, j] + r3 * σ_r[4, j]
            r_dot_σi = r1 * σ_i[2, j] + r2 * σ_i[3, j] + r3 * σ_i[4, j]
            C_r[i] += filter * (zr * r_dot_σr - zi * r_dot_σi)
            C_i[i] += filter * (zi * r_dot_σr + zr * r_dot_σi)
            zr = inv(d) * c
            zi = inv(d) * s
            C_r[i] += filter * (zr * σ_r[1, j] - zi * σ_i[1, j])
            C_i[i] += filter * (zi * σ_r[1, j] + zr * σ_i[1, j])
        end
    end
    return C
end

#TODO: vectorized version of the gradient operators
