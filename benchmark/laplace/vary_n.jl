##
import IFGF
using LinearAlgebra
using StaticArrays
using FMM3D

# parameters
pde = IFGF.Laplace(; dim = 3)
T = Float64
tol = 1e-4
test_fmm = false

nn = [10_000 * 4^n for n in 0:2]
# loop number of points
for n in nn
    println("n = $n")
    targets = sources = IFGF.points_on_unit_sphere(n)
    Xpts = Ypts = reinterpret(SVector{3,Float64}, targets)
    I = rand(1:n, 1000) |> unique
    charges = randn(n)
    GC.gc()
    tifgf_assemble = @elapsed A =
        IFGF.plan_laplace3d(T.(targets), T.(sources); charges = T.(charges), tol)
    println("|-- IFGF assemble: $tifgf_assemble")
    tifgf_prod = @elapsed pot_ifgf = IFGF.laplace3d(A; charges = T.(charges))
    println("|-- IFGF prod:     $tifgf_prod")
    println("|-- p:             $(A.p)")
    println("|-- h:             $(A.h)")
    exa = zeros(n)
    K = A.kernel
    IFGF.near_interaction_naive!(exa, K, Xpts, Ypts, charges, I, 1:n)
    er_ifgf = norm(pot_ifgf[I] - exa[I], 2) / norm(exa[I], 2)
    println("|-- IFGF er:       $er_ifgf")
    # cleanup and do fmm
    A = nothing
    GC.gc()
    if test_fmm
        tfmm_tot = @elapsed out = lfmm3d(tol, sources; charges, pg = 1)
        er_fmm = norm(out.pot[I] - exa[I], 2) / norm(exa[I], 2)
        println("|-- FMM tot:       $tfmm_tot")
        println("|-- FMM er:        $er_fmm")
    end
end
