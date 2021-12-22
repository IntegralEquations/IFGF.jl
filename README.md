# IFGF.jl

*Interpolated Factored Greens Function method*

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

This package provides an implementation of *Interpolated Factored Greens
Function* method (as originally described in [this
paper](https://arxiv.org/abs/2010.02857)) for accelerating the evaluation of
integral operators arising in boundary integral equation methods. Its main
structure is the `IFGFOperator`, which allows for *fast* evaluation of the
forward-map by interpolating the underlying kernel-density product at
well-chosen interpolation points.

Although the *IFGF* algorithm relies mostly on interpolation, the following
kernel-specific information is required:
- a `centered-factor`...

To illustrate its usage, let us consider the compression of the Helmholtz
free-space Greens function for a set of `100_000` points distributed on a sphere:

```julia
using IFGF, LinearAlgebra, StaticArrays
# sample some points on a sphere
m = 100_000
X = Y = [Point3D(sin(θ)cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for (θ,ϕ) in zip(π*rand(m),2π*rand(m))]
# define the kernel function
const k = 1
function K(x,y) 
    d = norm(x-y) + 1e-8
    exp(im*k*d)/d
end
```

The `IFGFOperator` allows to approximate the vector `y` given by

```julia
y = [sum(K(X[i],Y[j])*σ[j] for j in 1:m) for i in 1:m]
```

where `σ` is a given vector. The *IFGF* method is not kernel independent, so
some you must overload some methods for your kernel `K` in orprovide 

The simplest way to achieve this is by calling the
`assemble_ifgf(K,X,Y;kwargs...)`, where the following optional `kwargs` are may
be passed:

- `rtol`:
- `atol`:
- `threads`:
- `wavenumber`: the wavenumber `k` of the 



```julia

```







