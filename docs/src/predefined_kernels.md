```@meta
CurrentModule = IFGF
```

# [Predefined kernels](@id predefined-kernels-section)

While it is certainly possible to define your own kernel, doing so efficiently
can be non-trivial and time-consuming. *IFGF.jl* provides a
few predefined kernels that are commonly used in practice. If you need a kernel
that is not listed here, or if you have implemented a custom kernel and would
like for it to be added to the library, please open an issue and/or PR on the
[GitHub repository](https://github.com/IntegralEquations/IFGF.jl).

The kernel-specific *API* is heavily inspired by the [FMM3D
library](https://fmm3d.readthedocs.io), and provides functions to efficiently
compute sums of the form

```math
u(\boldsymbol{x}_i) = \sum_{j=1}^N G(\boldsymbol{x}_i - \boldsymbol{y}_j) c_j +\gamma_{1,\boldsymbol{y}} G(\boldsymbol{x}_i - \boldsymbol{y}_j) v_j,
```

where $\boldsymbol{x_i}$ are the target locations, $\boldsymbol{y}_j$ are the
source locations, $G(\boldsymbol{x} - \boldsymbol{y})$ is (up to a constant
factor) the fundamental solution of a given partial differential operator,
$\gamma_{1,\boldsymbol{y}}$ its (generalized) Neumann trace, and $c_i,v_i$ are
input vectors of the appropriate size. *IFGF.jl* computes the sum above in
log-linear complexity.

Each predefined kernel comes with an associated `plan` function that precomputes
the necessary data structure required to efficiently apply the forward map. If
you need to compute the forward map multiple times with the same kernel, it is a
good idea to build a *plan*.

In the following, we provide a brief overview of the predefined kernels.

!!! tip "Single-precision"
    All predefined kernels support single-precision arithmetic. If you do not
    need double-precision, you can save memory and computation time by using
    single-precision to represent your data.

## Laplace 3D operator

```@docs
IFGF.laplace3d
```

Forward map:

```@example laplace-3d
using IFGF
m = 100_000
targets = sources = rand(3, m)
charges = rand(m)
dipvecs = rand(3,m)
IFGF.laplace3d(targets, sources; charges, tol=1e-4)
```

Plan and forward map:

```@example laplace-3d
L = IFGF.plan_laplace3d(targets, sources; charges, dipvecs, tol=1e-4)
out  = zero(charges)
IFGF.laplace3d!(out, L; charges, dipvecs)
```

## Helmholtz 3D operator

```@docs
IFGF.helmholtz3d
```

Forward map:

```@example helmholtz-3d
using IFGF
m = 100_000
targets = sources = rand(3, m)
charges = rand(ComplexF64,m)
dipvecs = rand(ComplexF64,3,m)
k = 2π
IFGF.helmholtz3d(k, targets, sources; charges, tol=1e-4)
```

Plan and forward map:

```@example helmholtz-3d
L = IFGF.plan_helmholtz3d(k, targets, sources; charges, dipvecs, tol=1e-6)
out  = zero(charges)
IFGF.helmholtz3d!(out, L; charges, dipvecs)
```

## Stokes 3D operator

!!! warning 
    Stokes 3D operator should be considered experimental. Please report any
    issues you encounter.
    
```@docs
IFGF.stokes3d
```

Forward map:

```@example stokes-3d
using IFGF
m = 100_000
targets = sources = rand(3, m)
stoklet  = rand(3,m)
strslet = nothing
IFGF.stokes3d(targets, sources; stoklet, strslet, tol=1e-4)
```

Plan and forward map:

```@example stokes-3d
L = IFGF.plan_stokes3d(targets, sources; stoklet, strslet, tol=1e-4)
out = zero(stoklet)
IFGF.stokes3d!(out, L; stoklet)
```