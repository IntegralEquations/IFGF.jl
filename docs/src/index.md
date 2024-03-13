```@meta
CurrentModule = IFGF
```

# [IFGF.jl](@id home-section)

*A Julia implementation of the Interpolated Factored Green Function (IFGF) Method
 on adaptive trees*

## [Overview](@id overview-section)

This package implements an algorithm for approximating the forward map (i.e. the
matrix-vector product) of certain kernel matrices, where **kernel matrix** will
be used to refer to any matrix ``A \in \mathbb{F}^{m\times n}`` whose entries
can be computed from the following three components:

- a vector of target elements ``X = \left\{ \boldsymbol{x}_i \right\}_{i=1}^m`` 
- a vector of source elements ``Y = \left\{ \boldsymbol{y}_i \right\}_{i=1}^n`` 
- a kernel function ``K(x,y) : X \times Y \to \mathbb{F}``, where ``\mathbb{F}``
  is the return type of the kernel (and of the underlying matrix)

The entries of a kernel matrix are given by the explicit formula ``A_{i,j} =
K(\boldsymbol{x}_i,\boldsymbol{y}_j)``. Typically, ``X`` and ``Y`` are vectors
of points in ``\mathbb{R}^d``, and ``K`` is a function mapping elements of ``X``
and ``Y`` into ``\mathbb{R}`` or ``\mathbb{C}``.

Provided ``K`` has sufficient structure, an approximation of the product ``Ax``
can be computed in linear or log-linear complexity by various well known
algorithms such as the *Fast Multipole Method* or *hierarchical matrices*; this
package implements another acceleration algorithm termed *Interpolated Factored
Green Function (IFGF)* method.

*IFGF.jl* comes loaded with a few predefined kernels arising in mathematical
physics; check out the [predefined kernels](@ref predefined-kernels-section)
section for more details. Otherwise, setting up your own kernel is
straightforward, as shown next.

## Simple example

To illustrate how to set up the `IFGFOp`, let ``X`` and ``Y`` be a set of random
points on the unit cube, and take ``K`` to be the Helmholtz Green function in
three dimensions:

```math
    K(\boldsymbol{x},\boldsymbol{y}) = \frac{e^{ik|\boldsymbol{x}-\boldsymbol{y}|}}{4\pi |\boldsymbol{x}-\boldsymbol{y}|}
```

Setting up the aforementioned kernel matrix can be done as follows:

```@example helmholtz-simple
using IFGF, LinearAlgebra, StaticArrays
import IFGF: wavenumber

# random points on a cube
const Point3D = SVector{3,Float64}
m = 100_000
X = Y = rand(Point3D,m)

# define a the kernel matrix
struct HelmholtzMatrix <: AbstractMatrix{ComplexF64}
    X::Vector{Point3D}
    Y::Vector{Point3D}
    k::Float64
end

# indicate that this is an ocillatory kernel with wavenumber `k`
wavenumber(A::HelmholtzMatrix) = A.k

# functor interface
function (K::HelmholtzMatrix)(x,y)
    k = wavenumber(K)
    d = norm(x-y)
    return (d!=0) * (exp(im*k*d)*inv(4*pi*d)) # skip x == y case
end

# abstract matrix interface
Base.size(::HelmholtzMatrix) = length(X), length(Y)
Base.getindex(A::HelmholtzMatrix,i::Int,j::Int) = A(A.X[i],A.Y[j])

# create the abstract matrix
k = 4π   
A = HelmholtzMatrix(X,Y,k)
```

Note that `A` is a lazy ``100000 \times 100000`` matrix which computes its
entries on demand; as such, it has a light memory footprint. Trying to
instantiate the underlying `Matrix{ComplexF64}` would most likely result in you
running out of memory.

To build an approximation of `A`, simply do:

```@example helmholtz-simple
L = assemble_ifgf(A,X,Y; tol = 1e-4)
```

Check the documentation of [`assemble_ifgf`](@ref) for more information on the
available options.

You can now use `L` in lieu of `A` to approximate the matrix vector product, as
illustrated below:

```@example helmholtz-simple
x     = randn(ComplexF64,m)
y     = L*x
```

Note that `L`, while not as compact as `A`, still has a
relatively light memory footprint:

```@example helmholtz-simple
gb = Base.summarysize(L) / 1e9
print("Size of L: $gb gigabytes")
```

The quality of the approximation may be verified by computing the exact value at
a few randomly selected set of indices:

```@example helmholtz-simple
I     = rand(1:m,100)
exact = [sum(A[i,j]*x[j] for j in 1:m) for i in I] # matrix-vector product
er    = norm(y[I]-exact) / norm(exact) # relative error
@assert er < 1e-4 # hide
print("approximate relative error: $er")
```

To keep things simple, we neglected some important optimizations that can be
performed when defining your own kernel. See the [custom kernels](@ref
custom-kernel-section) for more information.
