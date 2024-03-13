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

charges = rand(ComplexF64, n)
dipvecs = rand(ComplexF64, 3, n)

I = rand(1:n, 1000)

@testset "Helmholtz3D" begin
    k = 2π
    pde = IFGF.Helmholtz(; dim = 3, k = 2π)

    @testset "single layer" begin
        K = IFGF.SingleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * charges[j] for j in 1:n) for i in I]
        for T in (Float32, Float64)
            y = IFGF.helmholtz3d(T(k), T.(X), T.(Y); charges = Complex{T}.(charges), p, h)
            @assert eltype(y) == Complex{T}
            @test norm(exa - y[I]) / norm(exa) < 1e-4
        end
    end

    @testset "double layer" begin
        K = IFGF.DoubleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * dipvecs[:, j] for j in 1:n) for i in I]
        for T in (Float32, Float64)
            y = IFGF.helmholtz3d(T(k), T.(X), T.(Y); dipvecs = Complex{T}.(dipvecs), p, h)
            @assert eltype(y) == Complex{T}
            @test norm(exa - y[I]) / norm(exa) < 1e-4
        end
    end

    @testset "adjoint double layer" begin
        K = IFGF.GradSingleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * charges[j] for j in 1:n) for i in I]
        exa = reinterpret(reshape, ComplexF64, exa) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.helmholtz3d(
                T(k),
                T.(X),
                T.(Y);
                charges = Complex{T}.(charges),
                grad = true,
                p,
                h,
            )
            @assert eltype(y) == Complex{T}
            @assert size(y, 1) == 3
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end

    @testset "Hessian single layer" begin
        K = IFGF.HessianSingleLayerKernel(pde)
        exa = [sum(K(Xpts[i], Ypts[j]) * dipvecs[:, j] for j in 1:n) for i in I]
        exa = reinterpret(reshape, ComplexF64, exa) # convert to a 3×n matrix
        for T in (Float32, Float64)
            y = IFGF.helmholtz3d(
                T(k),
                T.(X),
                T.(Y);
                dipvecs = Complex{T}.(dipvecs),
                grad = true,
                p,
                h,
            )
            @assert eltype(y) == Complex{T}
            @assert size(y, 1) == 3
            @test norm(exa - y[:, I]) / norm(exa) < 1e-4
        end
    end

    @testset "combined field" begin
        K = IFGF.CombinedFieldKernel(pde)
        exa =
            [sum(K(Xpts[i], Ypts[j]) * [charges[j]; dipvecs[:, j]] for j in 1:n) for i in I]
        for T in (Float32, Float64)
            y = IFGF.helmholtz3d(
                T(k),
                T.(X),
                T.(Y);
                dipvecs = Complex{T}.(dipvecs),
                charges = Complex{T}.(charges),
                p,
                h,
            )
            @test y ≈
                  IFGF.helmholtz3d(
                T(k),
                T.(X),
                T.(Y);
                dipvecs = Complex{T}.(dipvecs),
                p,
                h,
            ) + IFGF.helmholtz3d(T(k), T.(X), T.(Y); charges = Complex{T}.(charges), p, h)
            @assert eltype(y) == Complex{T}
            @test norm(exa - y[I]) / norm(exa) < 1e-4
        end
    end

    @testset "gradient combined field" begin
        #TODO: implement
    end
end
