using Test
import IFGF
using StaticArrays

# recursively check that all point in a cluster tree are in the bounding box
function test_cluster_tree(clt)
    bbox = IFGF.container(clt)
    for iloc in IFGF.index_range(clt)
        x = IFGF.root_elements(clt)[iloc]
        IFGF.center(x) ∈ bbox || (return false)
    end
    if !IFGF.isroot(clt)
        clt ∈ clt.parent.children || (return false)
    end
    if !IFGF.isleaf(clt)
        for child in clt.children
            test_cluster_tree(child) || (return false)
        end
    end
    return true
end

@testset "ClusterTree" begin
    @testset "1d" begin
        points = SVector.([4, 3, 1, 2, 5, -1.0])
        splitter = IFGF.DyadicSplitter(; nmax = 1)
        clt = IFGF.ClusterTree(points, splitter)
        @test sortperm(points) == clt.loc2glob
        splitter = IFGF.DyadicMinimalSplitter(; nmax = 1)
        clt = IFGF.ClusterTree(points, splitter)
        @test sortperm(points) == clt.loc2glob
    end

    @testset "2d" begin
        points = rand(SVector{2,Float64}, 1000)
        splitter = IFGF.DyadicSplitter(; nmax = 32)
        clt = IFGF.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
        splitter = IFGF.DyadicMinimalSplitter(; nmax = 1)
        clt = IFGF.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
    end

    @testset "3d" begin
        points = rand(SVector{3,Float64}, 1000)
        splitter = IFGF.DyadicSplitter(; nmax = 32)
        clt = IFGF.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
        splitter = IFGF.DyadicMinimalSplitter(; nmax = 1)
        clt = IFGF.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
    end

    @testset "Points and normals" begin
        points = rand(SVector{3,Float64}, 1000)
        els = [(; coords = points[i], normal = points[i]) for i in 1:1000]
        IFGF.center(x::eltype(els)) = x.coords
        splitter = IFGF.DyadicSplitter(; nmax = 32)
        clt = IFGF.ClusterTree(els, splitter)
        @test test_cluster_tree(clt)
        splitter = IFGF.DyadicMinimalSplitter(; nmax = 32)
        clt = IFGF.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
    end
end
