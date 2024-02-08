```@meta
CurrentModule = IFGF
```

# [IFGF.jl](@id home-section)

*A Julia implementation of the Interpolated Factored Green Function Method
(IFGF) on adaptive trees*

## [Introduction](@id introduction-section)

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
of points in ``\mathbb{R}^d``, and ``K`` is a function mapping pairs of points
into ``\mathbb{R}`` or ``\mathbb{C}``.

Provided ``K`` has sufficient structure, an approximation of the product ``Ax``
can be computed in linear or log-linear complexity by various well known
algorithms such as the *Fast Multipole Method* or *hierarchical matrices*; this
package implements another acceleration algorithm named *Interpolated Factored
Green Function* method.

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
m,n = 100_000, 100_000
X,Y = rand(Point3D,m), rand(Point3D,n)

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
    exp(im*k*d)*inv(4*pi*d)
end

# abstract matrix interface
Base.size(::HelmholtzMatrix) = length(X), length(Y)
Base.getindex(A::HelmholtzMatrix,i::Int,j::Int) = A(A.X[i],A.Y[j])

# create the abstract matrix
k = 8π   
A = HelmholtzMatrix(X,Y,k)
```

Note that `A` is a lazy ``100000 \times 100000`` matrix which computes its
entries on demand; as such, it has a light memory footprint. Trying to
instantiate the underlying `Matrix{ComplexF64}` would most likely result in you
running out of memory.

To build an approximation of `A` using the `IFGFOp`, you may use the high-level
constructor [`assemble_ifgf`](@ref) as follows:

```@example helmholtz-simple
L = assemble_ifgf(A,X,Y; tol = 1e-4)
```

Check the documentation of [`assemble_ifgf`](@ref) for more information on the
available options. One important point to keep in mind is that heuristics are
used to determine the optimal number of interpolation points given a tolerance
`tol`; if you want to control the number of interpolation points manually, you
may pass a `p` argument to the `assemble_ifgf` constructor instead.

!!! info
    To approximate oscillatory kernels, the `IFGF` algorithm adapts to the
    acoustic size of each of the cells in the source tree in order to keep the
    interpolation error constant across all levels of the tree. In practice,
    this means that memory and runtime will depend on both the total number of
    points `n` and on the wavenumber `k`, with smaller `k` being "easier" to
    approximate.

We can use `L` in lieu of `A` to approximate the matrix vector product, as
illustrated below:

```@example helmholtz-simple
x     = randn(ComplexF64,n)
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
exact = [sum(A[i,j]*x[j] for j in 1:n) for i in I] # matrix-vector product
er    = norm(y[I]-exact) / norm(exact) # relative error
print("approximate relative error: $er")
```

!!! tip "Kernel optimizations"
    To keep things simple in this example, we neglected some important
    optimizations that can be performed when defining your own kernel. In
    particular, you can overload some default (generic) methods for your own
    kernel to provide e.g. a vectorized evaluation using `SIMD` instructions. We
    also avoided the possible issue of kernel blowing up at ``\boldsymbol{x} =
    \boldsymbol{y}``.

## [Custom kernels] (@id custom-kernel-section)

When defining your own kernels, there is one required as well as some optional
methods to overload in order to provide kernel-specific information to the
constructor. The table below summarizes the interface methods, where `Y` is a
`SourceTree` object, `x` is an `SVector` representing a point, and `K` is your custom
kernel:

| **Method's name**                          | **Required** | **Brief description**                                                                  |
| ------------------------------------------ | ------------ | -------------------------------------------------------------------------------------- |
| [`wavenumber(K)`](@ref)                    | Yes          | Return the wavenumber of your kernel                                                   |
| `return_type(K)`                           | No           | Type of element returned by `K(x,y)`                                                   |
| [`centered_factor(K,x,Y)`](@ref)           | No           | A representative value of `K(x,y)` for `y ∈ Y`                                         |
| [`inv_centered_factor(K,x,Y)`](@ref)       | No           | A representative value of `inv(K(x,y))` for `y ∈ Y`                                    |
| [`transfer_factor(K,x,Y)`](@ref)           | No           | Transfer function given by `inv_centered_factor(K,x,parent(Y))*centered_factor(K,x,Y)` |
| [`near_interaction!(C,K,X,Y,σ,I,J)`](@ref) | No           | Compute `C[i] <- C[i] + ∑ⱼ K(X[i],Y[j])*σ[j]` for `i ∈ I`, `j ∈ J`                     |

Because the `IFGF` algorithms must adapt its interpolation scheme depending on
the frequency of oscillations to control the interpolation error, you must
extend the `IFGF.wavenumber` method. If the kernel is not oscillatory (e.g.
exponential kernel, Newtonian potential), simply return `0`.

The `centered_factor`, `inv_centered_factor`, and `transfer_factor` have
reasonable defaults, and typically are overloaded only for performance reasons
(e.g. if some analytic simplifications are available).

Finally, the `near_interaction` method is called at the leaves of the source
tree to compute the near interactions. Extending this function to make
use of e.g. SIMD instructions can significantly speed up some parts of the code.

## Advanced usage
