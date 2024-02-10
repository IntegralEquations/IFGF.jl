using LoopVectorization  # for automatically vectorizing some typical kernels
using StaticArrays

const Point3D = SVector{3,Float64}

# define some typical kernels
struct HelmholtzKernel
    k::Float64
end

function (K::HelmholtzKernel)(x, y)
    d = norm(x - y)
    v = exp(im * K.k * d) / (4 * π * d)
    return (!iszero(d)) * v
end

IFGF.wavenumber(K::HelmholtzKernel) = K.k

function IFGF.near_interaction!(C, K::HelmholtzKernel, X, Y, σ, I, J)
    Tx = eltype(X)
    Ty = eltype(Y)
    @assert Tx <: SVector && Ty <: SVector
    Xm = reshape(reinterpret(eltype(Tx), X), 3, :)
    Ym = reshape(reinterpret(eltype(Ty), Y), 3, :)
    @views helmholtz3d_sl_vec!(C[I], Xm[:, I], Ym[:, J], σ[J], K.k)
end

function IFGF.transfer_factor(K::HelmholtzKernel, x, Y)
    yc = IFGF.center(Y)
    yp = IFGF.center(IFGF.parent(Y))
    d  = norm(x - yc)
    dp = norm(x - yp)
    return exp(im * K.k * (d - dp)) * dp / d
end

function helmholtz3d_sl_vec!(C, X, Y, σ, k)
    m, n = size(X, 2), size(Y, 2)
    C_T = reinterpret(Float64, C)
    C_r = @views C_T[1:2:end, :]
    C_i = @views C_T[2:2:end, :]
    σ_T = reinterpret(Float64, σ)
    σ_r = @views σ_T[1:2:end, :]
    σ_i = @views σ_T[2:2:end, :]
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1, i] - Y[1, j])^2
            d2 += (X[2, i] - Y[2, j])^2
            d2 += (X[3, i] - Y[3, j])^2
            d = sqrt(d2)
            s, c = sincos(k * d)
            zr = inv(4π * d) * c
            zi = inv(4π * d) * s
            C_r[i] += (!iszero(d)) * (zr * σ_r[j] - zi * σ_i[j])
            C_i[i] += (!iszero(d)) * (zi * σ_r[j] + zr * σ_i[j])
        end
    end
    return C
end

struct LaplaceKernel end

function (K::LaplaceKernel)(x, y)
    d = norm(x - y)
    v = 1 / (4 * π * d)
    return (!iszero(d)) * v
end

IFGF.wavenumber(K::LaplaceKernel) = 0

function IFGF.near_interaction!(C, K::LaplaceKernel, X, Y, σ, I, J)
    Xm = reshape(reinterpret(Float64, X), 3, :)
    Ym = reshape(reinterpret(Float64, Y), 3, :)
    @views laplace3d_sl_vec!(C[I], Xm[:, I], Ym[:, J], σ[J])
end

function laplace3d_sl_vec!(C, X, Y, σ)
    m, n = size(X, 2), size(Y, 2)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1, i] - Y[1, j])^2
            d2 += (X[2, i] - Y[2, j])^2
            d2 += (X[3, i] - Y[3, j])^2
            # fast invsqrt code taken from here
            # https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
            invd = @fastmath Float64(1 / sqrt(Float32(d2)))
            invd = 1.5invd - 0.5d2 * invd * (invd * invd)
            C[i] += (!iszero(d2)) * (inv(4π) * invd * σ[j])
            # C[i] += inv(4π*sqrt(d2))*σ[j] # significalty slower
        end
    end
    return C
end

struct MaxwellKernel
    k::Float64
end

function (K::MaxwellKernel)(x, y)
    r = x - y
    d = norm(r)
    # helmholtz greens function
    g   = exp(im * K.k * d) / (4π * d)
    gp  = im * K.k * g - g / d
    gpp = im * K.k * gp - gp / d + g / d^2
    RRT = r * transpose(r) # rvec ⊗ rvecᵗ
    G   = g * LinearAlgebra.I + 1 / K.k^2 * (gp / d * LinearAlgebra.I + (gpp / d^2 - gp / d^3) * RRT)
    return (!iszero(d)) * G
end

IFGF.wavenumber(K::MaxwellKernel) = K.k

# function IFGF.centered_factor(K::MaxwellKernel,x,Y)
#     yc = IFGF.center(Y)
#     r  = x-yc
#     d = norm(r)
#     g   = exp(im*k*d)/(4π*d)
#     return g    # works fine!
# end

# function IFGF.transfer_factor(K::HelmholtzKernel,x,Y)
#     yc  = IFGF.center(Y)
#     yp  = IFGF.center(IFGF.parent(Y))
#     d   = norm(x-yc)
#     dp  = norm(x-yp)
#     exp(im*K.k*(d-dp))*dp/d
# end
