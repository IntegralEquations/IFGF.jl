##
using IFGF
using LinearAlgebra
using StaticArrays
using ParametricSurfaces
using Random
using Nystrom
Random.seed!(1)

const k = 16π
λ       = 2π/k
ppw     = 20
dx      = λ/ppw

clear_entities!()
geo = ParametricSurfaces.Sphere(;radius=1)
Ω   = Domain(geo)
Γ   = boundary(Ω)
np  = ceil(1.6/dx)
M   = meshgen(Γ,(np,np))
msh = NystromMesh(M,Γ;order=1)
Xpts = qcoords(msh) |> collect
Ypts = Xpts
nx = length(Xpts)
ny = length(Ypts)
@info nx,ny

function K(x,y)
    d = norm(x-y)
    if d == 0
        zero(ComplexF64)
    else
        ComplexF64(exp(im*k*norm(x-y))/norm(x-y))
    end
end

I   = rand(1:nx,1000)
B   = rand(ComplexF64,ny)
tfull = @elapsed exa = [sum(K(Xpts[i],Ypts[j])*B[j] for j in 1:ny) for i in I]
@info "Estimated time for full product: $(tfull*nx/1000)"

# trees
# spl   = CardinalitySplitter(;nmax=100)
# spl   = GeometricMinimalSplitter(;nmax=100)
# spl   = GeometricSplitter(;nmax=100)
spl = DyadicSplitter(;nmax=100)

function ds_oscillatory(source)
    # h    = radius(source.bounding_box)
    bbox = source.bounding_box
    w = bbox.high_corner - bbox.low_corner |> maximum
    ds   = Float64.((1,π/2,π/2))
    if k*w/2 < 1
        return ds
    else
        return ds ./ (k*w/2)
    end
end

# cone list
p = (3,5,5)
source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=ComplexF64)
target = initialize_target_tree(;points=Xpts,splitter=spl)
compute_interaction_list!(target,source,IFGF.admissible)
@info p
ds = (source) -> ds_oscillatory(source)
@hprofile compute_cone_list!(source,p,ds)
C  = zeros(ComplexF64,nx)
A = IFGFOperator(K,target,source)
@hprofile mul!(C,A,B)
er = norm(C[I]-exa,2) / norm(exa,2)
@info er,nx
