##
using IFGF
using LinearAlgebra
using StaticArrays
using FMM3D

includet(joinpath(IFGF.PROJECT_ROOT, "test", "simple_geometries.jl"))

# parameters
p = (5, 7, 7)
h = nothing
T = Float64
r = 1
nn = [10_000 * 4^n for n in 0:4]
# loop number of points
for n in nn
    pde = IFGF.Stokes(; dim = 3)
    K = IFGF.SingleLayerKernel(pde)
    X = Y = sphere_uniform(n, r)
    sources = reinterpret(reshape, Float64, X) |> collect
    I = rand(1:n, 100)
    charges = randn(T, 3, n)
    B = reinterpret(reshape, SVector{3,T}, charges) |> collect
    C = zero(B)
    exa = [sum(K(X[i], Y[j]) * B[j] for j in 1:n) for i in I]
    tifgf_assemble = @elapsed A = assemble_ifgf(K, X, Y; p)
    tifgf_prod = @elapsed mul!(C, A, B, 1, 0)
    er_ifgf = norm(C[I] - exa, 2) / norm(exa, 2)
    tfmm_tot = @elapsed out = stfmm3d(1e-3, sources; stoklet = charges, ppreg = 1)
    er_fmm = norm(2 * reinterpret(SVector{3,Float64}, out.pot)[I] - exa) / norm(exa, 2)
    println("n = $n")
    println("|-- IFGF assemble: $tifgf_assemble")
    println("|-- IFGF prod:     $tifgf_prod")
    println("|-- FMM tot:       $tfmm_tot")
    println("|-- FMM er:        $er_fmm")
    println("|-- IFGF er:       $er_ifgf")
end
