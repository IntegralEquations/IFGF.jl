# NOTE: currently we depend on LoopVectorization.jl to power the SIMD magic. As
# it seems that the package may be deprecated, this file also contains some WIP
# "manual" vectorization routines which uses only the SIMD.jl package


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
