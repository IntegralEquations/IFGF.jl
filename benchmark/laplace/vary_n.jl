##
using IFGF
using LinearAlgebra
using StaticArrays
using FMM3D

IFGF.use_minimal_conedomain(false)
IFGF.usethreads(true)

includet(joinpath(IFGF.PROJECT_ROOT, "test", "simple_geometries.jl"))

# parameters
p = (8, 8, 8)
h = 1.0
pde = IFGF.Laplace(; dim = 3)
K = IFGF.SingleLayerKernel(pde)
# K = LaplaceKernel()
T = Float64

nn = [10_000 * 4^n for n in 4:4]
# loop number of points
for n in nn
    Xpts = Ypts = sphere_uniform(n, 1)
    sources = reinterpret(reshape, Float64, Xpts) |> collect
    I = rand(1:n, 100)
    B = randn(T, n)
    charges = B
    C = zeros(T, n)
    exa = [sum(K(Xpts[i], Ypts[j]) * B[j] for j in 1:n) for i in I]
    tifgf_assemble = @elapsed A = assemble_ifgf(K, Xpts, Ypts; p, h)
    tifgf_prod = @elapsed mul!(C, A, B, 1, 0)
    tfmm_tot = @elapsed out = lfmm3d(1e-5, sources; charges, pg = 1)
    er_ifgf = norm(C[I] - exa, 2) / norm(exa, 2)
    er_fmm = norm(out.pot[I] - exa, 2) / norm(exa, 2)
    println("n = $n")
    println("|-- IFGF assemble: $tifgf_assemble")
    println("|-- IFGF prod:     $tifgf_prod")
    println("|-- FMM tot:       $tfmm_tot")
    println("|-- FMM er:        $er_fmm")
    println("|-- IFGF er:       $er_ifgf")
end
