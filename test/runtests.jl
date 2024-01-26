using SafeTestsets

@safetestset "Chebyshev interpolation" begin
    include("chebinterp_test.jl")
end
@safetestset "IFGF operator" begin
    include("ifgfoperator_test.jl")
end
