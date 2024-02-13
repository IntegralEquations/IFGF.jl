using Test
using IFGF
using StaticArrays
import IFGF

for F in (Float32, Float64)
    for T in (F, Complex{F})
        for dim in 1:3
            # a polynomial of sufficiently low order which should be exactly
            # reproduced by the cheb interpolant
            f = (x) -> T(prod(x)^2 + sum(x)prod(x) + 1)
            @testset "T=$T, dim=$dim" begin
                # test the chebyshev transform
                p    = ntuple(i -> 5 + i, dim)
                x̂   = IFGF.chebnodes(p, F)
                vals = f.(x̂)
                c1   = IFGF.chebtransform_fftw!(copy(vals))
                c2   = IFGF.chebtransform_native!(copy(vals))
                @test eltype(c1) == eltype(c2) == T
                @test c1 ≈ c2
                # test the chebyshev evaluation
                x = @SVector rand(F, dim)
                rec =
                    IFGF.HyperRectangle(ntuple(i -> -F(1.0), dim), ntuple(i -> F(1.0), dim))
                SZ = Val(p)
                @test IFGF.chebeval(c1, x, rec, SZ) ≈ f(x)
            end
        end
    end
end
