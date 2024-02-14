##
using IFGF
using LinearAlgebra
using StaticArrays
using FMM3D

includet(joinpath(IFGF.PROJECT_ROOT, "test", "simple_geometries.jl"))

# parameters
p = (4, 6, 6)
T = ComplexF64
ppw = 22.4 # point per wavelength
r = 1
area = 4π * r^2
nn = [10_000 * 4^n for n in 0:3]
fixed_k = false # either fix k and increase n, or increase k with n
# loop number of points
for n in nn
    k = fixed_k ? 2π : sqrt(n / ppw^2 / area * 4π^2)
    pde = IFGF.Helmholtz(; k, dim = 3)
    K = IFGF.SingleLayerKernel(pde)
    X = Y = sphere_uniform(n, r)
    sources = reinterpret(reshape, Float64, X) |> collect
    I = rand(1:n, 100)
    charges = randn(T, n)
    B = charges
    C = zeros(T, n)
    exa = [sum(K(X[i], Y[j]) * B[j] for j in 1:n) for i in I]
    tifgf_assemble = @elapsed A = assemble_ifgf(K, X, Y; p)
    tifgf_prod = @elapsed mul!(C, A, B, 1, 0)
    er_ifgf = norm(C[I] - exa, 2) / norm(exa, 2)
    tfmm_tot = @elapsed out = hfmm3d(1e-3, k, sources; charges, pg = 1)
    er_fmm = norm(out.pot[I] - exa, 2) / norm(exa, 2)
    println("n = $n")
    println("|-- k:             $(trunc(k,digits=2))")
    println("|-- IFGF assemble: $tifgf_assemble")
    println("|-- IFGF prod:     $tifgf_prod")
    println("|-- FMM tot:       $tfmm_tot")
    println("|-- FMM er:        $er_fmm")
    println("|-- IFGF er:       $er_ifgf")
end
