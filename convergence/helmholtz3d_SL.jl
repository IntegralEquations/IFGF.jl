##
using IFGF
using LinearAlgebra
using StaticArrays
using ParametricSurfaces
using Random
using Nystrom
using FMM3D
Random.seed!(1)

includet(joinpath(IFGF.PROJECT_ROOT,"test","testutils.jl"))

const k = 8π
λ       = 2π/k
ppw     = 16[]
dx      = λ/ppw

struct HelmholtzKernel
    k::Float64
end

@fastmath function (K::HelmholtzKernel)(x,y)
    d = norm(x-y)
    return exp(im*K.k*d)/(4*π*d)
end

function IFGF.near_interaction!(C,K::HelmholtzKernel,X,Y,σ,I,J)
    Xm = reshape(reinterpret(Float64,X),3,:)
    Ym = reshape(reinterpret(Float64,Y),3,:)
    @views _helmholtz3d_sl_fast!(C[I],Xm[:,I],Ym[:,J],σ[J],K.k)
end

const T = ComplexF64

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

K = HelmholtzKernel(k)

I   = rand(1:nx,20)
B   = randn(T,ny)
tfull = @elapsed exa = [sum(i == j ? 0*im : K( Xpts[i],Ypts[j])*B[j] for j in 1:ny) for i in I]
@info "Estimated time for full product: $(tfull*nx/20)"

# trees
splitter = DyadicSplitter(;nmax=200)

# cone

p = (3,5,5)
ds_func = IFGF.cone_domain_size_func(k,(1.0,π/2,π/2))
A = IFGFOperator(K,Ypts,Xpts;splitter,p,ds_func)
C = zeros(T,nx)
# for _ in 1:4
#     t = @elapsed mul!(C,A,B,1,0;threads=true)
#     er = norm(C[I]-exa,2) / norm(exa,2)
#     @info "" er,nx,t
# end

@hprofile mul!(C,A,B,1,0;threads=false)
er = norm(C[I]-exa,2) / norm(exa,2)
@info "" er,nx
@info Base.summarysize(A)/1e9
##

t = @elapsed mul!(C,A,B,1,0;threads=true)
er = norm(C[I]-exa,2) / norm(exa,2)
@info "" er,nx,t


##

xx = [coords(x) for x in Xpts]
sources = reinterpret(Float64,xx)
sources = reshape(sources,3,:)  |> collect

charges = reinterpret(ComplexF64,B) |> collect

t = @elapsed vals = hfmm3d(1e-3,ComplexF64(k),sources;charges,pg=1)

er = norm(vals.pot[I] ./ 4π - exa,2) / norm(exa,2)

@info er,nx,t
