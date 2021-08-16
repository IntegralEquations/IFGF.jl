##
using IFGF
using LinearAlgebra
using StaticArrays
using ParametricSurfaces
using Random
using Nystrom
Random.seed!(1)

const k = 8π
λ       = 2π/k
ppw     = 16
dx      = λ/ppw

pde = Elastostatic(dim=3,μ=1,λ=2)
K   = SingleLayerKernel(pde)

function IFGF.centered_factor(K::typeof(K),x,yc)
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
Xdofs = msh.dofs
Ydofs = Xdofs
Xpts = qcoords(msh) |> collect
Ypts = Xpts
nx = length(Xpts)
ny = length(Ypts)
@info nx,ny

I   = rand(1:nx,1000)
B   = rand(T,ny)
tfull = @elapsed exa = [sum(K(Xdofs[i],Ydofs[j])*B[j] for j in 1:ny) for i in I]
@info "Estimated time for full product: $(tfull*nx/1000)"

# trees
spl = DyadicSplitter(;nmax=100)

function ds_oscillatory(source)
    # h    = radius(source.bounding_box)
    bbox = source.bounding_box
    w = bbox.high_corner - bbox.low_corner |> maximum
    ds   = Float64.((1.0,π/2,π/2))
    δ    = k*w/2
    if δ < Inf
        return ds
    else
        return ds ./ δ
    end
end

# cone list
p = (node) -> (3,5,5)
source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=T)
permute!(Ydofs,source.loc2glob)
target = initialize_target_tree(;points=Xpts,splitter=spl)
# permute!(Xdofs,target.loc2glob)
compute_interaction_list!(target,source,IFGF.admissible)
ds = (source) -> ds_oscillatory(source)
@hprofile compute_cone_list!(source,p,ds)
@info source.data.p
C  = zeros(T,nx)
A = IFGFOperator(K,target,source,Xdofs,Ydofs)
@hprofile mul!(C,A,B)
er = norm(C[I]-exa,2) / norm(exa,2)
@info er,nx
