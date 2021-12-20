##
using IFGF
using LinearAlgebra
using StaticArrays
using Random
Random.seed!(1)

includet(joinpath(IFGF.PROJECT_ROOT,"test","testutils.jl"))

# k    = 48π
# k    = 100
k    = 10
λ    = 2π/k
ppw  = 10
dx   = λ/ppw
r    = 1
area = 4*π*r^2
N    = ceil(Int,area/dx^2)
N    = 100_000

struct HelmholtzKernel
    k::Float64
end

function (K::HelmholtzKernel)(x,y)
    d = norm(x-y)
    v = exp(im*K.k*d)/(4*π*d)
    return (!iszero(d))*v
end

function IFGF.near_interaction!(C,K::HelmholtzKernel,X,Y,σ,I,J)
    Xm = reshape(reinterpret(Float64,X),3,:)
    Ym = reshape(reinterpret(Float64,Y),3,:)
    @views _helmholtz3d_sl_fast!(C[I],Xm[:,I],Ym[:,J],σ[J],K.k)
end

const T = ComplexF64

XX   = fibonacci(N,r) |> transpose
Xpts = reinterpret(SVector{3,Float64},XX) |> vec |> collect
Ypts = Xpts
nx = length(Xpts)
ny = length(Ypts)
@info "" nx,ny

K = HelmholtzKernel(k)

I   = rand(1:nx,20)
B   = 2*randn(T,ny) .- 1
tfull = @elapsed exa = [sum(i == j ? 0*im : K( Xpts[i],Ypts[j])*B[j] for j in 1:ny) for i in I]
@info "Estimated time for full product: $(tfull*nx/20)"

# trees
splitter = CardinalitySplitter(;nmax=200)
# splitter = DyadicSplitter(;nmax=200)

# cone
ds_func = IFGF.cone_domain_size_func(k,(1.0,π/2,π/2)./8)
p = (4,4,4)
tp = @elapsed A = IFGFOp(K,Ypts,Xpts;splitter,p,ds_func,lite=true,threads=true)
# tp = @elapsed A = IFGFOp(K,Ypts,Xpts;splitter,tol=1e-8,ds_func,lite=true,threads=true)
C = zeros(T,nx)

t = @elapsed mul!(C,A,B,1,0;threads=true)
# @hprofile mul!(C,A,B,1,0;threads=false)
er = norm(C[I]-exa,2) / norm(exa,2)
@info "" er,nx,tp,t, tp+t
@info Base.summarysize(A)/1e9
##

# mul!(C,A,B,1,0;threads=true)
# t = @elapsed mul!(C,A,B,1,0;threads=true)
# er = norm(C[I]-exa,2) / norm(exa,2)
# @info "" er,nx,t
