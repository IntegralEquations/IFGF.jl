using Test
using IFGF
using LinearAlgebra
using StaticArrays

n = 2000
X = Y = IFGF.points_on_unit_sphere(n, Float64)
Xf = Yf = Float32.(X)
Xpts = Ypts = reinterpret(SVector{3,Float64}, X)
p = 6
h = 1

stoklet = rand(3, n)
strslet = rand(6, n)

I = rand(1:n, 1000)

@testset "Stokes3D" begin
    pde = IFGF.Stokes(; dim = 3)

    @testset "single layer" begin
        K = IFGF.SingleLayerKernel(pde)
        exa_svec = [sum(K(Xpts[i], Ypts[j]) * stoklet[:, j] for j in 1:n) for i in I]
        exa = reinterpret(reshape, Float64, exa_svec) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.stokes3d(T.(X), T.(Y); stoklet = T.(stoklet), p, h)
            @assert eltype(y) == T
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end

    @testset "double layer" begin
        K = IFGF.DoubleLayerKernel(pde)
        exa_svec = [sum(K(Xpts[i], Ypts[j]) * strslet[:, j] for j in 1:n) for i in I]
        exa = reinterpret(reshape, Float64, exa_svec) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.stokes3d(T.(X), T.(Y); strslet = T.(strslet), p, h)
            @assert eltype(y) == T
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end

    @testset "combined field" begin
        K = IFGF.CombinedFieldKernel(pde)
        exa_svec = [
            sum(K(Xpts[i], Ypts[j]) * [stoklet[:, j]; strslet[:, j]] for j in 1:n) for
            i in I
        ]
        exa = reinterpret(reshape, Float64, exa_svec) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.stokes3d(
                T.(X),
                T.(Y);
                stoklet = T.(stoklet),
                strslet = T.(strslet),
                p,
                h,
            )
            @assert eltype(y) == T
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end
end
