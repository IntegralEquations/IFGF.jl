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

charges = rand(n)
dipvecs = rand(3, n)

I = rand(1:n, 1000)

@testset "Laplace3D" begin
    pde = IFGF.Laplace(; dim = 3)

    @testset "single layer" begin
        K = IFGF.SingleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * charges[j] for j in 1:n) for i in I]
        for T in (Float32, Float64)
            y = IFGF.laplace3d(T.(X), T.(Y); charges = T.(charges), p, h)
            @assert eltype(y) == T
            @test norm(exa - y[I]) / norm(exa) < 1e-4
        end
    end

    @testset "double layer" begin
        K = IFGF.DoubleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * dipvecs[:, j] for j in 1:n) for i in I]
        for T in (Float32, Float64)
            y = IFGF.laplace3d(T.(X), T.(Y); dipvecs = T.(dipvecs), p, h)
            @assert eltype(y) == T
            @test norm(exa - y[I]) / norm(exa) < 1e-4
        end
    end

    @testset "adjoint double layer" begin
        K = IFGF.GradSingleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * charges[j] for j in 1:n) for i in I]
        exa = reinterpret(reshape, Float64, exa) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.laplace3d(T.(X), T.(Y); charges = T.(charges), grad = true, p, h)
            @assert eltype(y) == T
            @assert size(y, 1) == 3
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end

    @testset "Hessian single layer" begin
        K = IFGF.HessianSingleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * dipvecs[:, j] for j in 1:n) for i in I]
        exa = reinterpret(reshape, Float64, exa) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.laplace3d(T.(X), T.(Y); dipvecs = T.(dipvecs), grad = true, p, h)
            @assert eltype(y) == T
            @assert size(y, 1) == 3
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end

    @testset "combined field" begin
        K = IFGF.CombinedFieldKernel(pde)
        exa =
            [sum(K(Xpts[i], Ypts[j]) * [charges[j]; dipvecs[:, j]] for j in 1:n) for i in I]
        for T in (Float32, Float64)
            y = IFGF.laplace3d(
                T.(X),
                T.(Y);
                dipvecs = T.(dipvecs),
                charges = T.(charges),
                p,
                h,
            )
            @assert eltype(y) == T
            @test norm(exa - y[I]) / norm(exa) < 1e-4
        end
    end

    @testset "gradient combined field" begin
        #TODO: implement
    end
end
