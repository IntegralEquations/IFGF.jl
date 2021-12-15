##
using IFGF
using LinearAlgebra
using StaticArrays
using ParametricSurfaces
using Random
using Nystrom
Random.seed!(1)

includet(joinpath(IFGF.PROJECT_ROOT,"test","testutils.jl"))

k    = 64π
λ    = 2π/k
ppw  = 8
dx   = λ/ppw

struct LaplaceKernel
end

function (K::LaplaceKernel)(x,y)
    d = norm(x-y)
    v = 1/(4*π*d)
    return (!iszero(d))*v
end

function IFGF.near_interaction!(C,K::LaplaceKernel,X,Y,σ,I,J)
    Xm = reshape(reinterpret(Float64,X),3,:)
    Ym = reshape(reinterpret(Float64,Y),3,:)
    @views _laplace3d_sl_fast!(C[I],Xm[:,I],Ym[:,J],σ[J])
end

const T = Float64

clear_entities!()
geo = ParametricSurfaces.Sphere(;radius=1)
Ω   = Domain(geo)
Γ   = boundary(Ω)
np  = ceil(Int,2/dx)
M   = ParametricSurfaces.meshgen(Γ,(np,np))
msh = NystromMesh(M,Γ;order=1)
Xpts = [coords(dof) for dof in msh.dofs]
Ypts = Xpts
nx = length(Xpts)
ny = length(Ypts)
@info "" nx,ny

K = LaplaceKernel()

I   = rand(1:nx,20)
B   = randn(T,ny)
tfull = @elapsed exa = [sum(i == j ? 0 : K( Xpts[i],Ypts[j])*B[j] for j in 1:ny) for i in I]
@info "Estimated time for full product: $(tfull*nx/20)"

# trees
splitter = CardinalitySplitter(;nmax=200)
# splitter = DyadicSplitter(;nmax=200)

# cone
p = (4,5,5)
ds_func = IFGF.cone_domain_size_func(0,(1.0,π/2,π/2))
tp = @elapsed A = IFGFOp(K,Ypts,Xpts;splitter,p,ds_func,lite=true,threads=true)
# @hprofile A = IFGFOp(K,Ypts,Xpts;splitter,p,ds_func)
C = zeros(T,nx)

# @hprofile mul!(C,A,B,1,0;threads=true)
# er = norm(C[I]-exa,2) / norm(exa,2)
# @info "" er,nx
# @info Base.summarysize(A)/1e9
##

t = @elapsed mul!(C,A,B,1,0;threads=true)
er = norm(C[I]-exa,2) / norm(exa,2)
@info "" er,nx,tp,t
