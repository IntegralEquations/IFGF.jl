using Test
using IFGF
using StaticArrays
using IFGF: chebeval_novec, chebeval_vec, chebeval_vec, chebtransform_fftw!, chebtransform_native!, HyperRectangle

for T in (Float64, ComplexF64)
    for dim in 1:3

        @testset "Cheyshev transform: T=$T, dim=$dim" begin
            vals = rand(T,ntuple(i->5+i,dim))
            c1   = chebtransform_fftw!(vals)
            c2   = chebtransform_native!(vals)
            @test c1 ≈ c2
        end

        @testset "Clenshaw summation: T=$T, dim=$dim" begin
            sz   = ntuple(i->3+i,dim)
            c    = rand(T,prod(sz))
            x     = @SVector rand(dim)
            rec = HyperRectangle(ntuple(i->-1.0,dim),ntuple(i->1.0,dim))
            SZ = Val(sz)
            v1 = chebeval_novec(c,x,rec,SZ)
            v2 = chebeval_vec(c,x,rec,SZ)
            @test v1 ≈ v2
        end

    end
end
