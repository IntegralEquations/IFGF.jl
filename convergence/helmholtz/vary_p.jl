##
using IFGF
using LinearAlgebra
using StaticArrays
using Random
using Plots
plotlyjs()
Random.seed!(1)

includet(joinpath(IFGF.PROJECT_ROOT,"test","testutils.jl"))

k    = 8π
λ    = 2π/k
ppw  = 10
dx   = λ/ppw

Xpts = Ypts = sphere_uniform_surface_mesh(dx)
npts = length(Xpts)

K    = HelmholtzKernel(k)
T    = ComplexF64

I   = rand(1:npts,100)
B   = randn(T,npts)
exa = [sum(K( Xpts[i],Ypts[j])*B[j] for j in 1:npts) for i in I]

# trees
splitter   = Trees.CardinalitySplitter(;nmax=200)

# cone size function
ds_func = IFGF.cone_domain_size_func(k,(1.0,π/2,π/2))

# loop over interpolation order
pvec = 2:8
er   = []
for p in pvec
    ptuple  = (p,p,p)
    A       = IFGFOp(K,Xpts,Ypts;splitter,p=ptuple,ds_func,threads=true)
    C       = zeros(T,npts)
    mul!(C,A,B,1,0;threads=true)
    ee = norm(C[I]-exa,2) / norm(exa,2)
    push!(er,ee)
end

print(
    "Uniform points on surfaces of sphere with parameters:
        r   = 1
        k   = $k
        n   = $npts
        ppw = $ppw"
)


fig = plot(pvec,er,yscale=:log10,m=:circle,
    ylabel="approximate ℓ₂ relative error",
    xlabel="p",
    label="sphere")

dir = @__DIR__
savefig(fig,joinpath(dir,"sphere_p_convergence.png"))
