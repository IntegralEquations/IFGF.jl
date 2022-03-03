##
using IFGF
using LinearAlgebra
using StaticArrays
using Random

include(joinpath(IFGF.PROJECT_ROOT,"test","testutils.jl"))

# parameters
p        = (3,5,5)
ppw      = 10 # point per wavelength
T        = ComplexF64
r        = 1 # sphere radius
area     = 4π*r^2
nmax     = 100
Δs       = (0.5,π/2,π/2)

# loop over wavenumbers
kvec = [2π,2π,4π,8π]
# kvec = [2π,2π,4π,8π,16π,32π,64π]

for k in kvec
    n        = ceil(Int,ppw^2*k^2*area/(4π^2))
    K        = HelmholtzKernel(k)
    Xpts = Ypts = sphere_uniform(n,r)
    I   = collect(1:1000)
    B   = ones(T,n)
    C   = zeros(T,n)
    exa = [sum(K( Xpts[i],Ypts[j])*B[j] for j in 1:n) for i in I]
    ta  = @elapsed A = assemble_ifgf(K,Xpts,Ypts;nmax,order=p.-1,Δs,threads=true)
    tp = @elapsed mul!(C,A,B,1,0;threads=true)
    er = norm(C[I]-exa,2) / norm(exa,2)
    println("="^80)
    println(A)
    println("lambda=$(2*pi/k),N=$n,checksum=$(exa[1]),tassemble = $ta, tprod=$tp,
    error=$er")
end
