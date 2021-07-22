using Test
using LinearAlgebra
using StaticArrays
using IFGF
using IFGF: cone_index, interp2cart, cart2interp

@testset "Near field" begin
    @testset "Single leaf" begin
        nx,ny = 100, 200
        Xpts   = rand(SVector{2,Float64},nx)
        Ypts   = rand(SVector{2,Float64},ny)
        spl   = DyadicSplitter(;nmax=250)
        source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=Float64)
        target = initialize_target_tree(;points=Xpts,splitter=spl)
        compute_interaction_list!(target,source,IFGF.admissible)
        # cone list
        p = (4,4)
        ds_func = x -> (1/4,2π/4)
        compute_cone_list!(source,p,ds_func)
        K(x,y) = 1/norm(x-y)
        A_mat = [K(x,y) for x in target.points, y in source.points]
        B     = rand(ny)
        C     = zeros(nx)
        A = IFGFOperator(K,target,source)
        mul!(C,A,B)
        @test C ≈ A_mat*B
    end
    @testset "Tree" begin
        nx,ny = 100, 200
        Xpts   = rand(SVector{2,Float64},nx)
        Ypts   = rand(SVector{2,Float64},ny)
        spl    = DyadicSplitter(;nmax=100)
        source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=Float64)
        target = initialize_target_tree(;points=Xpts,splitter=spl)
        compute_interaction_list!(target,source,IFGF.admissible)
        # cone list
        p = (4,4)
        ds_func = x -> (1/4,2π/4)
        compute_cone_list!(source,p,ds_func)
        K(x,y) = 1/norm(x-y)
        A_mat = [K(x,y) for x in target.points, y in source.points]
        B     = rand(ny)
        C     = zeros(nx)
        A = IFGFOperator(K,target,source)
        mul!(C,A,B)
        @test C ≈ A_mat*B
    end
end

@testset "Far field leaf" begin
    @testset "Single leaf" begin
        nx,ny = 100, 200
        Xpts   = rand(SVector{2,Float64},nx)
        Ypts   = [SVector(10,10)+rand(SVector{2,Float64}) for _ in 1:ny]
        spl   = DyadicSplitter(;nmax=250)
        source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=Float64)
        target = initialize_target_tree(;points=Xpts,splitter=spl)
        compute_interaction_list!(target,source,IFGF.admissible)
        # cone list
        p = (10,10)
        ds_func = x -> (1/4,2π/8)
        compute_cone_list!(source,p,ds_func)
        K(x,y) = 1/norm(x-y)
        A_mat = [K(x,y) for x in target.points, y in source.points]
        B     = rand(ny)
        C     = zeros(nx)
        A = IFGFOperator(K,target,source)
        clear_interpolants!(source)
        mul!(C,A,B)
        @test C ≈  A_mat*B
    end
    @testset "Tree" begin
        nx,ny = 100, 200
        Xpts   = rand(SVector{2,Float64},nx)
        Ypts   = [ SVector(10,10)+rand(SVector{2,Float64}) for _ in 1:ny]
        spl   = DyadicSplitter(;nmax=100)
        source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=Float64)
        target = initialize_target_tree(;points=Xpts,splitter=spl)
        compute_interaction_list!(target,source,IFGF.admissible)
        # cone list
        p = (10,10)
        ds_func = x -> (1/4,2π/8)
        compute_cone_list!(source,p,ds_func)
        @test length(source.data.far_list) == 1
        @test length(source.data.near_list) == 0
        @test length(source.children[1].data.far_list) == 0
        @test length(source.children[1].data.near_list) == 0
        K(x,y) = 1/norm(x-y)
        A_mat = [K(x,y) for x in target.points, y in source.points]
        B     = rand(ny)
        C     = zeros(nx)
        A     = IFGFOperator(K,target,source)
        clear_interpolants!(source)
        mul!(C,A,B)
        @test C ≈  A_mat*B
    end
end

@testset "Near and far field" begin
    nx,ny  = 10000, 10000
    Xpts   = rand(SVector{2,Float64},nx)
    Ypts   = [rand(SVector{2,Float64}) for _ in 1:ny]
    spl   = DyadicSplitter(;nmax=100)
    source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=Float64)
    target = initialize_target_tree(;points=Xpts,splitter=spl)
    compute_interaction_list!(target,source,IFGF.admissible)
    # cone list
    p = (10,10)
    ds_func = x -> (1/4,2π/8)
    compute_cone_list!(source,p,ds_func)
    K(x,y) = 1/norm(x-y)
    A_mat = [K(x,y) for x in target.points, y in source.points]
    B     = rand(ny)
    C     = zeros(nx)
    A = IFGFOperator(K,target,source)
    clear_interpolants!(source)
    mul!(C,A,B)
    C - A_mat*B
    @test C ≈  A_mat*B
end
