##
using IFGF
using LinearAlgebra
using StaticArrays
using FMM3D

# parameters
F = Float64
T = Complex{F}
tol = 1e-4
ppw = 10 # point per wavelength
area = 4π # surface area of unit sphere for computing k
fixed_k = false # either fix k and increase n, or increase k with n
test_fmm = true

nn = [10_000 * 4^n for n in 0:6]
# loop number of points
for n in nn
    k = fixed_k ? 2π : sqrt(n / ppw^2 / area * (2π)^2)
    pde = IFGF.Helmholtz(; k, dim = 3)
    K = IFGF.SingleLayerKernel(pde)
    println("n = $n")
    println("|-- k:             $(trunc(k,digits=2))")
    targets = sources = IFGF.points_on_unit_sphere(n)
    Xpts = Ypts = reinterpret(SVector{3,Float64}, targets)
    I = rand(1:n, 1000) |> unique
    charges = randn(ComplexF64, n)
    tifgf_assemble = @elapsed A = IFGF.plan_helmholtz3d(
        F.(k),
        F.(targets),
        F.(sources);
        charges = T.(charges),
        tol,
    )
    println("|-- IFGF assemble: $tifgf_assemble")
    println("|-- p:             $(A.p)")
    println("|-- h:             $(A.h)")
    tifgf_prod = @elapsed pot_ifgf = IFGF.helmholtz3d(A; charges = T.(charges))
    println("|-- IFGF prod:     $tifgf_prod")
    exa = zeros(ComplexF64, n)
    IFGF.near_interaction_naive!(exa, K, Xpts, Ypts, charges, I, 1:n)
    er_ifgf = norm(pot_ifgf[I] - exa[I], 2) / norm(exa[I], 2)
    println("|-- IFGF er:       $er_ifgf")
    # cleanup and test fmm
    A = nothing
    GC.gc()
    if test_fmm
        tfmm_tot = @elapsed out = hfmm3d(tol, k, sources; charges, pg = 1)
        println("|-- FMM tot:       $tfmm_tot")
        er_fmm = norm(out.pot[I] - exa[I], 2) / norm(exa[I], 2)
        println("|-- FMM er:        $er_fmm")
    end
end
