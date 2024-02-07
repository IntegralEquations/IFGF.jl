using Test
using LinearAlgebra
using StaticArrays
using IFGF

K(x, y) = 1 / norm(x - y)
IFGF.wavenumber(::typeof(K)) = 0

@testset "Near field" begin
    @testset "Single leaf" begin
        nx, ny = 100, 200
        nz = 5
        Xpts = rand(SVector{2,Float64}, nx)
        Ypts = rand(SVector{2,Float64}, ny)
        p = (4, 4)
        ds_func = (x) -> (1 / 4, 2π / 4)
        A_mat = [K(x, y) for x in Xpts, y in Ypts]
        B = rand(ny)
        B_mat = rand(ny, nz)
        C = zeros(nx)
        A = assemble_ifgf(K, Xpts, Ypts; p, h = 1, nmax = 250)
        mul!(C, A, B, 1, 0)
        @test size(A) == (nx, ny)
        @test C ≈ A_mat * B
        # @test A*B_mat ≈ A_mat*B_mat
    end
    @testset "Tree" begin
        nx, ny = 100, 200
        p      = (4, 4)
        nz     = 5
        Xpts   = rand(SVector{2,Float64}, nx)
        Ypts   = rand(SVector{2,Float64}, ny)
        A_mat  = [K(x, y) for x in Xpts, y in Ypts]
        B      = rand(ny)
        B_mat  = rand(ny, nz)
        C      = zeros(nx)
        A      = assemble_ifgf(K, Xpts, Ypts; p, h = 1, nmax = 100)
        mul!(C, A, B, 1, 0)
        @test size(A) == (nx, ny)
        @test C ≈ A_mat * B
        # @test A*B_mat ≈ A_mat*B_mat
    end
end

@testset "Far field leaf" begin
    @testset "Single leaf" begin
        nx, ny = 100, 200
        nz     = 5
        p      = (12, 12)
        Xpts   = rand(SVector{2,Float64}, nx)
        Ypts   = [SVector(10, 10) + rand(SVector{2,Float64}) for _ in 1:ny]
        A_mat  = [K(x, y) for x in Xpts, y in Ypts]
        B      = rand(ny)
        B_mat  = rand(ny, nz)
        C      = zeros(nx)
        A      = assemble_ifgf(K, Xpts, Ypts; p, h = 1, nmax = 250)
        mul!(C, A, B, 1, 0)
        @test size(A) == (nx, ny)
        @test C ≈ A_mat * B
        # @test A*B_mat ≈ A_mat*B_mat
    end
    @testset "Tree" begin
        nx, ny = 100, 200
        nz     = 5
        p      = (12, 12)
        Xpts   = rand(SVector{2,Float64}, nx)
        Ypts   = [SVector(10, 10) + rand(SVector{2,Float64}) for _ in 1:ny]
        A_mat  = [K(x, y) for x in Xpts, y in Ypts]
        B      = rand(ny)
        B_mat  = rand(ny, nz)
        C      = zeros(nx)
        A      = assemble_ifgf(K, Xpts, Ypts; p, h = 1, nmax = 100)
        mul!(C, A, B, 1, 0)
        @test size(A) == (nx, ny)
        @test C ≈ A_mat * B
    end
end

@testset "Near and far field" begin
    tol = 1e-4
    nx, ny = 1000, 1000
    nz     = 3
    Xpts   = rand(SVector{3,Float64}, nx)
    Ypts   = [SVector(1, 1, 1) + rand(SVector{3,Float64}) for _ in 1:ny]
    A_mat  = [K(x, y) for x in Xpts, y in Ypts]
    B      = rand(ny)
    exact = A_mat*B
    B_mat  = rand(ny, nz)
    C      = zeros(nx)
    A      = assemble_ifgf(K, Xpts, Ypts; tol)
    mul!(C, A, B, 1, 0)
    @test norm(C - exact) / norm(exact) < tol
    @test size(A) == (nx, ny)
    bytes_ifgf = Base.summarysize(A) # to make sure the size of A does not change upon further products
    @test C == mul!(C, A, B, 1, 0) # recompute to verify that the result does not change
    @test bytes_ifgf == Base.summarysize(A)
end
