##
using IFGF
using LinearAlgebra
using StaticArrays
using Random
using Plots
plotlyjs()
Random.seed!(1)

includet(joinpath(IFGF.PROJECT_ROOT,"test","testutils.jl"))

# parameters
p        = (4,4,4)
ppw      = 20
splitter = Trees.DyadicSplitter(;nmax=100)
T        = ComplexF64

# specify the function to generate the point clouds
pts_func = sphere_uniform_surface_mesh

# loop over wavenumbers
kvec = [2π,2π,4π,8π,16π,32π]
er   = []
ta   = [] # assemble time
tp   = [] # product time
npts = []
for k in kvec
    ds_func  = IFGF.cone_domain_size_func(k,(1.0,1.0,1.0))
    K    = HelmholtzKernel(k)
    λ    = 2π/k
    dx   = λ/ppw
    Xpts = Ypts = pts_func(dx)
    n    = length(Xpts)
    push!(npts,n)
    I   = rand(1:n,100)
    B   = randn(T,n)
    C   = zeros(T,n)
    exa = [sum(K( Xpts[i],Ypts[j])*B[j] for j in 1:n) for i in I]
    tmp = @elapsed A  = IFGFOp(K,Xpts,Ypts;splitter,p,ds_func,threads=true)
    push!(ta,tmp)
    tmp = @elapsed mul!(C,A,B,1,0;threads=true)
    push!(tp,tmp)
    ee = norm(C[I]-exa,2) / norm(exa,2)
    push!(er,ee)
end

fig = plot(npts[2:end],ta[2:end],m=:circle,
    ylabel="time (s)",
    xlabel="N",
    yscale=:log10,
    xscale=:log10,
    label="precomputation")
plot!(npts[2:end],tp[2:end],m=:cross,label="forward map")
# add a reference slop
p1 = 1
p2 = 2
b  = 4
yy = @. npts[2:end]^p1 * log(b,npts[2:end])^p2*tp[end]/(npts[end]^p1*log(b,npts[end])^p2)
plot!(npts[2:end],yy,ls=:dash,label="loglinear slope")

dir = @__DIR__
savefig(fig,joinpath(dir,"sphere_k_scaling.png"))
