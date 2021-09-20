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

pde = Helmholtz(dim=3,k=k)
G   = SingleLayerKernel(pde)
K   = DoubleLayerKernel(pde)

function IFGF.centered_factor(::typeof(K),x,yc)
    r = coords(x)-yc
    d = norm(r)
    exp(im*k*d)/d*(-im*k+1/d)
    # G(x,yc)
end

const T = return_type(K)

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

function ds_oscillatory(source)
    # h    = radius(source.bounding_box)
    bbox = IFGF.container(source)
    w = maximum(IFGF.high_corner(bbox)-IFGF.low_corner(bbox))
    ds   = Float64.((1.0,π/2,π/2))
    δ    = k*w/2
    if δ < 1
        return ds
    else
        return ds ./ δ
    end
end

# cone list
p  = (node) -> (3,5,5)
ds = (source) -> ds_oscillatory(source)
C  = zeros(T,nx)
A  = IFGFOperator(K,Ypts,Xpts;datatype=T,splitter,p_func,ds_func,_profile=true)
@hprofile mul!(C,A,B)
er = norm(C[I]-exa,2) / norm(exa,2)
@info er,nx
