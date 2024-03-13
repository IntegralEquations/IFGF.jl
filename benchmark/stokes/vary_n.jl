##
import IFGF
using LinearAlgebra
using StaticArrays
using FMM3D

# parameters
pde = IFGF.Stokes(; dim = 3)
K = IFGF.SingleLayerKernel(pde)
T = Float64
tol = 1e-3
test_fmm = true

nn = [10_000 * 4^n for n in 0:6]
# loop number of points
for n in nn
    println("n = $n")
    targets = sources = IFGF.points_on_unit_sphere(n)
    Xpts = Ypts = reinterpret(SVector{3,Float64}, targets)
    I = rand(1:n, 1000) |> unique
    stoklet = randn(3, n)
    GC.gc()
    tifgf_assemble = @elapsed A =
        IFGF.plan_stokes3d(T.(targets), T.(sources); stoklet = T.(stoklet), tol)
    println("|-- IFGF assemble: $tifgf_assemble")
    tifgf_prod = @elapsed pot_ifgf = IFGF.stokes3d(A; stoklet = T.(stoklet))
    println("|-- IFGF prod:     $tifgf_prod")
    println("|-- p:             $(A.p)")
    println("|-- h:             $(A.h)")
    exa = zeros(3, n)
    exa_ = IFGF._unsafe_wrap_vector_of_sarray(exa)
    stoklet_ = IFGF._unsafe_wrap_vector_of_sarray(stoklet)
    IFGF.near_interaction_naive!(exa_, K, Xpts, Ypts, stoklet_, I, 1:n)
    er_ifgf = norm(pot_ifgf[:, I] - exa[:, I], 2) / norm(exa[:, I], 2)
    println("|-- IFGF er:       $er_ifgf")
    # cleanup and do fmm
    A = nothing
    GC.gc()
    if test_fmm
        tfmm_tot = @elapsed out = stfmm3d(1tol, sources; stoklet, ppreg = 1)
        er_fmm = norm(out.pot[:, I] - exa[:, I]) / norm(exa[:, I], 2)
        println("|-- FMM tot:       $tfmm_tot")
        println("|-- FMM er:        $er_fmm")
    end
end
