using LinearAlgebra
using Random
using LoopVectorization
using StaticArrays
using BenchmarkTools
using IFGF
Random.seed!(1)
BLAS.set_num_threads(1)

function _helmholtz3d_sl_fast(C, X, Y, σ, k)
    m, n = size(X, 2), size(Y, 2)
    Cr, Ci = IFGF.real_and_imag(C)
    σr, σi = IFGF.real_and_imag(σ)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1, i] - Y[1, j])^2
            d2 += (X[2, i] - Y[2, j])^2
            d2 += (X[3, i] - Y[3, j])^2
            d = sqrt(d2)
            s, c = sincos(k * d)
            zr = inv(4π * d) * c
            zi = inv(4π * d) * s
            Cr[i] += (!iszero(d)) * (zr * σr[j] - zi * σi[j])
            Ci[i] += (!iszero(d)) * (zi * σr[j] + zr * σi[j])
        end
    end
    return C
end

T = Float64

nn = []
t1 = []
t2 = []
for k in 1:3
    m = n = 25 * 2^k
    C = zeros(Complex{T}, m)
    σ = rand(Complex{T}, n)
    X = Y = rand(T, 3, m)
    # Y = rand(T, 3, n)
    k = one(T)
    M = [
        (norm(x - y) != 0) * (exp(im * k * norm(x - y)) * inv(4π * norm(x - y))) for
        x in eachcol(X), y in eachcol(Y)
    ]
    er = _helmholtz3d_sl_fast(copy(C), X, Y, σ, k) - mul!(copy(C), M, σ) |> norm
    b1 = @belapsed _helmholtz3d_sl_fast($C, $X, $Y, $σ, $k)
    b2 = @belapsed mul!($C, $M, $σ)
    println("n = $n")
    println("|-- matrix free: $b1")
    println("|-- blas:        $b2")
    println("|-- ratio:       $(b1 / b2)")
    println("|-- error:       $er")
    println()
end
