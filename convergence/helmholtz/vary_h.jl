##
using IFGF
using LinearAlgebra
using StaticArrays
using Random
using Plots
plotlyjs()
Random.seed!(1)

includet(joinpath(IFGF.PROJECT_ROOT, "test", "testutils.jl"))

k   = 8π
λ   = 2π / k
ppw = 10
dx  = λ / ppw

Xpts = Ypts = sphere_uniform_surface_mesh(dx)
npts = length(Xpts)

K = HelmholtzKernel(k)
T = ComplexF64

I   = rand(1:npts, 100)
B   = randn(T, npts)
exa = [sum(K(Xpts[i], Ypts[j]) * B[j] for j in 1:npts) for i in I]

# trees
splitter = Trees.CardinalitySplitter(; nmax = 200)

# interpolation order
p = (3, 3, 3)

# loop over interpolation mesh sizes
hvec = 1:-0.1:0.2 |> collect
er   = []
for h in hvec
    ds_func = IFGF.cone_domain_size_func(k, h .* (1, 1, 1))
    A       = IFGFOp(K, Ypts, Xpts; splitter, p, ds_func, threads = true)
    C       = zeros(T, npts)
    mul!(C, A, B, 1, 0; threads = true)
    ee = norm(C[I] - exa, 2) / norm(exa, 2)
    push!(er, ee)
end

print("Uniform points on surfaces of sphere with parameters:
          r   = 1
          k   = $k
          n   = $npts
          ppw = $ppw")

fig = plot(
    hvec,
    er;
    yscale = :log10,
    xscale = :log10,
    m = :circle,
    ylabel = "approximate ℓ₂ relative error",
    xlabel = "h",
    label = "sphere",
)

dir = @__DIR__
savefig(fig, joinpath(dir, "sphere_h_convergence.png"))
