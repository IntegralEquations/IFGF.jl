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

pde = Maxwell(dim=3,k=k)
K   = SingleLayerKernel(pde)

function IFGF.centered_factor(K::typeof(K),x,yc)
    r = coords(x)-yc
    d = norm(r)
    g   = exp(im*k*d)/(4π*d)
    gp  = im*k*g - g/d
    gpp = im*k*gp - gp/d + g/d^2
    # RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    #return g + 1/k^2*(gp/d + gpp - gp/d)
    #return (1+im/(k*d)-1/(k*d)^2)*g    
    return g    # works fine!
end

const T = SVector{3,ComplexF64}

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
spl = DyadicSplitter(;nmax=100)

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
p = (node) -> (3,5,5)
source = initialize_source_tree(;points=Ypts,splitter=spl,datatype=T,Xdatatype=eltype(Xpts))
target = initialize_target_tree(;points=Xpts,splitter=spl)
compute_interaction_list!(target,source,IFGF.admissible)
#
ds = (source) -> ds_oscillatory(source)
@hprofile compute_cone_list!(source,p,ds)
@info source.data.p
C  = zeros(T,nx)
A = IFGFOperator(K,target,source)
@hprofile mul!(C,A,B)
er = norm(C[I]-exa,2) / norm(exa,2)
@info er,nx
