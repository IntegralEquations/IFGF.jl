using Test
using LinearAlgebra
using StaticArrays
using IFGF

@testset "Near field" begin
    @testset "Single leaf" begin
        nx,ny = 100, 200
        Xpts  = rand(SVector{2,Float64},nx)
        Ypts  = rand(SVector{2,Float64},ny)
        splitter = DyadicSplitter(;nmax=250)
        p_func   = (x) -> (4,4)
        ds_func  = (x) -> (1/4,2π/4)
        K(x,y)   = 1/norm(x-y)
        A_mat = [K(x,y) for x in Xpts, y in Ypts]
        B     = rand(ny)
        C     = zeros(nx)
        A     = IFGFOperator(K,Ypts,Xpts;datatype=Float64,splitter,p_func,ds_func)
        mul!(C,A,B)
        @test C ≈ A_mat*B
    end
    @testset "Tree" begin
        nx,ny = 100, 200
        Xpts  = rand(SVector{2,Float64},nx)
        Ypts  = rand(SVector{2,Float64},ny)
        splitter = DyadicSplitter(;nmax=100)
        p_func   = (x) -> (4,4)
        ds_func  = (x) -> (1/4,2π/4)
        K(x,y)   = 1/norm(x-y)
        A_mat = [K(x,y) for x in Xpts, y in Ypts]
        B     = rand(ny)
        C     = zeros(nx)
        A     = IFGFOperator(K,Ypts,Xpts;datatype=Float64,splitter,p_func,ds_func)
        mul!(C,A,B)
        @test C ≈ A_mat*B
    end
end

@testset "Far field leaf" begin
    @testset "Single leaf" begin
        nx,ny = 100, 200
        Xpts  = rand(SVector{2,Float64},nx)
        Ypts  = [SVector(10,10)+rand(SVector{2,Float64}) for _ in 1:ny]
        splitter = DyadicSplitter(;nmax=250)
        p_func   = (x) -> (10,10)
        ds_func  = (x) -> (1/4,2π/8)
        K(x,y)   = 1/norm(x-y)
        A_mat = [K(x,y) for x in Xpts, y in Ypts]
        B     = rand(ny)
        C     = zeros(nx)
        A     = IFGFOperator(K,Ypts,Xpts;datatype=Float64,splitter,p_func,ds_func)
        mul!(C,A,B)
        @test C ≈ A_mat*B
    end
    @testset "Tree" begin
        nx,ny = 100, 200
        Xpts  = rand(SVector{2,Float64},nx)
        Ypts  = [SVector(10,10)+rand(SVector{2,Float64}) for _ in 1:ny]
        splitter = DyadicSplitter(;nmax=100)
        p_func   = (x) -> (10,10)
        ds_func  = x -> (1/4,2π/8)
        K(x,y)   = 1/norm(x-y)
        A_mat = [K(x,y) for x in Xpts, y in Ypts]
        B     = rand(ny)
        C     = zeros(nx)
        A     = IFGFOperator(K,Ypts,Xpts;datatype=Float64,splitter,p_func,ds_func)
        mul!(C,A,B)
        @test C ≈ A_mat*B
    end
end

@testset "Near and far field" begin
    nx,ny  = 5000, 5000
    Xpts   = rand(SVector{2,Float64},nx)
    Ypts   = [rand(SVector{2,Float64}) for _ in 1:ny]
    splitter = DyadicSplitter(;nmax=20)
    p_func   =       x -> (10,10)
    ds_func  = x -> (1/4,2π/8)
    K(x,y)   = 1/norm(x-y)
    A_mat = [K(x,y) for x in Xpts, y in Ypts]
    B     = rand(ny)
    C     = zeros(nx)
    A     = IFGFOperator(K,Ypts,Xpts;datatype=Float64,splitter,p_func,ds_func)
    mul!(C,A,B)
    @test C ≈ A_mat*B
end
