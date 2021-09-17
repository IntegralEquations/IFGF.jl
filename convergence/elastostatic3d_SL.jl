##
using IFGF
using LinearAlgebra
using StaticArrays
using ParametricSurfaces
using Random
using Nystrom
Random.seed!(1)

const k = 4π
λ       = 2π/k
ppw     = 16
dx      = λ/ppw

pde = Elastostatic(dim=3,μ=1,λ=2)
K   = SingleLayerKernel(pde)

function IFGF.centered_factor(K::typeof(K),x,ysource::SourceTree)
    yc = center(ysource)
    r = coords(x)-yc
    d = norm(r)
    1/d
end

const T = SVector{3,Float64}

clear_entities!()
geo = ParametricSurfaces.Sphere(;radius=1)
Ω   = Domain(geo)
Γ   = boundary(Ω)
np  = ceil(2/dx)
M   = meshgen(Γ,(np,np))
msh = NystromMesh(M,Γ;order=1)
Xpts = msh.dofs
Ypts = Xpts
nx = length(Xpts)
ny = length(Ypts)
@info nx,ny

I   = rand(1:nx,1000)
B   = rand(T,ny)
tfull = @elapsed exa = [sum(K(Xpts[i],Ypts[j])*B[j] for j in 1:ny) for i in I]
@info "Estimated time for full product: $(tfull*nx/1000)"

# trees
splitter = DyadicSplitter(;nmax=100)

# cone list
p_func  = (node) -> (3,5,5)
ds_func = IFGF.cone_domain_size_func(nothing)
C  = zeros(T,nx)
A  = IFGFOperator(K,Ypts,Xpts;datatype=T,splitter,p_func,ds_func,_profile=true)
@hprofile mul!(C,A,B)
er = norm(C[I]-exa,2) / norm(exa,2)
@info er,nx
