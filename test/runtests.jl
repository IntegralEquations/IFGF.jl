using SafeTestsets

@safetestset "Chebyshev interpolation" begin
    include("chebinterp_test.jl")
end

@safetestset "Cluster trees" begin
    include("clustertree_test.jl")
end

@safetestset "IFGF operator" begin
    include("ifgfoperator_test.jl")
end

@safetestset "Kernels" begin
    include("laplace_test.jl")
    include("helmholtz_test.jl")
    include("stokes_test.jl")
end
