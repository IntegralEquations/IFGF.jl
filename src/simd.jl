# NOTE: currently we depend on LoopVectorization.jl to power the SIMD magic. As
# it seems that the package may be deprecated, this file also contains some WIP
# "manual" vectorization routines which uses only the SIMD.jl package

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
    C::AbstractVector{T},
    K::SingleLayerKernel{<:Stokes{3}},
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
            C[1, i] += filter * (σ[1, j] * invd + r1 * γ)
            C[2, i] += filter * (σ[2, j] * invd + r2 * γ)
            C[3, i] += filter * (σ[3, j] * invd + r3 * γ)
        end
    end
    return C
end

## SIMD.jl versions, commented for now

# @fastmath function IFGF.near_interaction!(
#     C::AbstractVector{T},
#     ::IFGF.SingleLayerKernel{IFGF.Laplace{3}},
#     Xp::VectorOfPoints{3,T},
#     Yp::VectorOfPoints{3,T},
#     σ::Vector{T},
#     I::UnitRange,
#     J::UnitRange,
# ) where {T}
#     VS = SIMD_BYTES ÷ sizeof(T)
#     X = Xp.data
#     Y = Yp.data
#     # we will split I and J into (Iv,Ir) and (Jv,Jr) where Iv and Jv are
#     # divisible by VS and Ir and Jr are the remainders indices. We then process
#     # the blocks using SIMD when possible
#     ilane = VecRange{VS}(0)
#     jlane = VecRange{VS}(0)

#     Iv, Ir = simd_split(I, VS)
#     Jv, Jr = simd_split(J, VS)

#     for i in Iv
#         iglob = ilane + i
#         x1, x2, x3 = X[iglob, 1], X[iglob, 2], X[iglob, 3]
#         for j in J
#             y1, y2, y3 = Y[j, 1], Y[j, 2], Y[j, 3]
#             c = σ[j]
#             v = _lap_sl3d(x1, x2, x3, y1, y2, y3, c)
#             C[iglob] += v
#         end
#     end

#     for j in Jv
#         jglob = jlane + j
#         y1, y2, y3 = Y[jglob, 1], Y[jglob, 2], Y[jglob, 3]
#         c = σ[jglob]
#         for i in Ir
#             x1, x2, x3 = X[i, 1], X[i, 2], X[i, 3]
#             v = _lap_sl3d(x1, x2, x3, y1, y2, y3, c)
#             C[i] += sum(v)
#         end
#     end

#     for i in Ir
#         x1, x2, x3 = X[i, 1], X[i, 2], X[i, 3]
#         for j in Jr
#             y1, y2, y3 = Y[j, 1], Y[j, 2], Y[j, 3]
#             c = σ[j]
#             v = _lap_sl3d(x1, x2, x3, y1, y2, y3, c)
#             C[i] += sum(v)
#         end
#     end
#     return C
# end

# @inline function _lap_sl3d(x1, x2, x3, y1, y2, y3, c)
#     d2 = (x1 - y1)^2 + (x2 - y2)^2 + (x3 - y3)^2
#     v = vifelse(d2 <= IFGF.SAME_POINT_TOLERANCE, zero(d2), inv(sqrt(d2)))
#     return v * c
# end
