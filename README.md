# IFGF.jl

*A Julia implementation of the [Interpolated Factored Green Function Method](https://arxiv.org/abs/2010.02857)*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://WaveProp.github.io/IFGF.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://WaveProp.github.io/IFGF.jl/dev)
[![Build Status](https://github.com/WaveProp/IFGF.jl/workflows/CI/badge.svg)](https://github.com/WaveProp/IFGF.jl/actions)
[![Coverage](https://codecov.io/gh/WaveProp/IFGF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/WaveProp/IFGF.jl)

## Installation
Install from the Pkg REPL:
```
pkg> add https://github.com/WaveProp/IFGF.jl
```

## Overview

This package provides an implementation of the [Interpolated Factored Green
Function Method (IFGF)](https://arxiv.org/abs/2010.02857) for accelerating the
evaluation of certain dense linear operators arising in boundary integral equation
methods. In particular, it provides an efficient approximation of the
matrix-vector product for both non-oscillatory kernels (e.g. Laplace, Stokes) and
oscillatory kernels (e.g. Helmholtz, Maxwell).

To illustrate how the `IFGFOp` can be used to approximate dense linear maps,
consider the problem of computing `Ax` where `A` is an `M×N` matrix with
entries given by `A[i,j] = K(X[i],Y[j])`, with `K(x,y)=exp(ik|x-y|)/|x-y|` being
the kernel function, and `X,Y` being a vector of `M` and `N` points in three
dimensions. The following code show how one may set up the aforementioned
matrix:

```julia
using IFGF, LinearAlgebra, StaticArrays
# random points on a cube
const Point3D = SVector{3,Float64}
m,n = 100_000, 100_000
X,Y = rand(Point3D,m), rand(Point3D,n)
# define the kernel function
struct HelmholtzMatrix <: AbstractMatrix{ComplexF64}
    X::Vector{Point3D}
    Y::Vector{Point3D}
    k::Float64
end

IFGF.wavenumber(A::HelmholtzMatrix) = A.k

# abstract matrix interface
Base.size(A::HelmholtzMatrix) = length(X), length(Y)
Base.getindex(A::HelmholtzMatrix,i::Int,j::Int) = A(A.X[i],A.Y[j])

# functor interface
function (K::HelmholtzMatrix)(x,y)
    k = IFGF.wavenumber(K)
    d = norm(x-y)
    exp(im*k*d)/d
end

# create the abstract matrix
A = HelmholtzMatrix(X,Y,1)
```

Although the memory footprint of the object `A` is very small (it *lazily*
computes its entries), multiplying it by a vector has complexity proportional to
`m*n`, which can be prohibitively costly for large problem sizes. The `IFGFOp`
computes an approximation as follows:

```julia
L = assemble_ifgf(A,X,Y;tol=1e-3)
mbytes = Base.summarysize(L) / 1e6
```

The structure `L` has a light memory footprint, occupying less than `20`
megabytes in this example. We can use it in lieu of `A` to approximate the
matrix vector product, as illustrated below:

```julia
x     = randn(ComplexF64,n)
y     = L*x
```

In order to assess the quality of the approximation, we may take a few random
rows and compare the approximate result against the exact one:

```julia
I  = rand(1:m,100)
exact = [sum(A[i,j]*x[j] for j in 1:n) for i in I]
er    = norm(y[I]-exact) / norm(exact) 
```

The error should be smaller than the prescribed tolerance.

See the [documentation](https://waveprop.github.io/IFGF.jl/dev/) for more
details and examples.
