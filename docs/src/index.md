```@meta
CurrentModule = IFGF
```

# [IFGF.jl](@id home-section)

*A pure Julia implementation of the Interpolated Factored Green Function Method
on adaptive trees*

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

The main structure exported by this package is the `IFGFOp{T}`, which inherits
from `AbstractMatrix{T}` and supports some basic linear algebra operations. To
construct an `IFGFOp` you will need to specify a kernel `K`, the target elements
`X`, and the source elements `Y`. You may also need to pass some
problem-specific information regarding `K`. 

To illustrate how to set up the `IFGFOp`, let ``X`` and ``Y`` be a
set of random points on the unit cube, and take ``K`` to be the Helmholtz Green
function in three dimensions:

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
Base.size(A::HelmholtzMatrix) = length(X), length(Y)
Base.getindex(A::HelmholtzMatrix,i::Int,j::Int) = A(A.X[i],A.Y[j])

# create the abstract matrix
k = 8π   
A = HelmholtzMatrix(X,Y,k)
```

Note that `A` is a lazy ``100000 \times 100000`` matrix which computes its entries on
demand; as such, it has a light memory footprint. Trying to instantiate the underlying
`Matrix{ComplexF64}` would most likely result in you running out of memory.

To build an approximation of `A` using the `IFGFOp`, you may use the high-level
constructor [`assemble_ifgf`](@ref) as follows:

```@example helmholtz-simple
L = assemble_ifgf(A,X,Y;tol=1e-3)
```

Check the documentation of [`assemble_ifgf`](@ref) for more information on the
available options. Note that `L`, while not as compact as `A`, still has a
relatively light memory footprint:

```@example helmholtz-simple
mbytes = Base.summarysize(L) / 1e6
print("Size of L: $mbytes megabytes")
```

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
    \boldsymbol{y}``. See the the
    [`test/testutils.jl`](https://github.com/WaveProp/IFGF.jl/blob/main/test/testutils.jl)
    file for an optimized definition of some classical kernels, or the [Custom
    kernels](@ref custom-kernel-section) section for a discussion on what
    methods can be overloaded.

## [Custom kernels] (@id custom-kernel-section)

When defining your own kernel matrix, there is one required as well as some
optional methods to overload in order to provide kernel-specific information to
the `IFGFOp` constructor. The table below summarizes the interface methods,
where `Y` is a `SourceTree`, `x` is an `SVector` representing a point, and `K`
is your custom kernel:

| **Method's name**                    | **Required** | **Brief description**                          |
| ------------------------------------ | ------------ | ---------------------------------------------- |
| [`wavenumber(K)`](@ref)              | Yes          | Return the wavenumber of your kernel           |
| `return_type(K)`| No | Type of element returned by `K(x,y)` |
| [`centered_factor(K,x,Y)`](@ref)     |  No      | A representative value of `K(x,y)` for `y ∈ Y` |
| [`inv_centered_factor(K,x,Y)`](@ref)     |  No      | A representative value of `inv(K(x,y))` for `y ∈ Y` |
| [`transfer_factor(K,x,Y)`](@ref)| No | Transfer function given by `inv_centered_factor(K,x,parent(Y))*centered_factor(K,x,Y)` |
| [`near_interaction!(C,K,X,Y,σ,I,J)`](@ref)| No | Compute `C[i] <- C[i] + ∑ⱼ K(X[i],Y[j])*σ[j]` for `i ∈ I`, `j ∈ J` |

Because the `IFGF` algorithms must adapt its interpolation scheme depending on
the frequency of oscillations to control the interpolation error, you must
extend the `IFGF.wavenumber` method. If the kernel is not oscillatory (e.g.
exponential kernel, Newtonian potential), simply return `0`.

The `return_type` method will be used to infer the type of the underlying
abstract matrix. By default, it will use `Base.promote_op` to try and infer a
return type, but you may force that type manually if needed by extending this method.

The `centered_factor`, `inv_centered_factor`, and `transfer_factor` have
reasonable defaults, and typically are overloaded only for performance reasons
(e.g. if some analytic simplifications are available).

Finally, the `near_interaction` method is called at the leaves of the source
tree to compute, well, the near interactions. Extending this function to make
use of e.g. SIMD instructions can significantly speed up some parts of the code.

To illustrate how to define a new kernel with the aforementioned hooks, we will
first extend the definition of the `HelmholtzMatrix` (done in the
[introduction](@ref introduction-section)) by implementing a vectorized
`near_interaction` method using the `LoopVectorization` package. The code below,
though more complex, implements a vectorized evaluation of the dense forward
map:

```@example helmholtz-simple
using LoopVectorization

function IFGF.near_interaction!(C,K::HelmholtzMatrix,X,Y,σ,I,J)
    Tx = eltype(X)
    Ty = eltype(Y)
    @assert Tx <: SVector && Ty <: SVector
    Xm = reshape(reinterpret(eltype(Tx),X),3,:)
    Ym = reshape(reinterpret(eltype(Ty),Y),3,:)
    @views helmholtz3d_sl_vec!(C[I],Xm[:,I],Ym[:,J],σ[J],K.k)
end

function helmholtz3d_sl_vec!(C::AbstractVector{Complex{T}},X,Y,σ,k) where {T}
    @assert eltype(σ) == eltype(C)
    m,n = size(X,2), size(Y,2)
    C_T = reinterpret(T, C)
    C_r = @views C_T[1:2:end,:]
    C_i = @views C_T[2:2:end,:]
    σ_T = reinterpret(T, σ)
    σ_r = @views σ_T[1:2:end,:]
    σ_i = @views σ_T[2:2:end,:]
    @turbo for j in 1:n # LoopVectorization magic
        for i in 1:m
            d2 = (X[1,i] - Y[1,j])^2
            d2 += (X[2,i] - Y[2,j])^2
            d2 += (X[3,i] - Y[3,j])^2
            d  = sqrt(d2)
            s, c = sincos(k * d)
            zr = inv(4π*d) * c
            zi = inv(4π*d) * s
            C_r[i] += (!iszero(d))*(zr*σ_r[j] - zi*σ_i[j])
            C_i[i] += (!iszero(d))*(zi*σ_r[j] + zr*σ_i[j])
        end
    end
    return C
end
```

The other optimization that we can perform for the `HelmhotlzMatrix` is to
rewrite the [`transfer_factor`](@ref) so that it uses only one exponential. The
default definition of `transfer_factor(K,x,Y)` simply divides `K(x,yc)` by `K(x,yp)`,
where `yc` is the center of the source box `Y` and `yp` is the center of the
parent of `Y`. This division can be simplified to avoid computing two complex
exponentials as follows:

```@example helmholtz-simple
function IFGF.transfer_factor(K::HelmholtzMatrix,x,Y)
    yc  = IFGF.center(Y)
    yp  = IFGF.center(IFGF.parent(Y))
    d   = norm(x-yc)
    dp  = norm(x-yp)
    exp(im*K.k*(d-dp))*dp/d
end
```

We can check that the result is still correct with the code below:

```@example helmholtz-simple
y     = L*x
er    = norm(y[I]-exact) / norm(exact) # relative error
print("approximate relative error: $er")
```

Of course, you should benchmark your code (with your specific kernel) to see if
and when it may be useful to provide faster versions of these methods.

!!! note "Tensor-valued kernels"
    Everything said so far applies if `K` returns a tensor instead of a scalar.
    This is the case e.g. for the dyadic Green function for time-harmonic
    Maxwell's equations, given by

    ```math
        K(\boldsymbol{x},\boldsymbol{y}) = \mathbb{G}(\boldsymbol{x},\boldsymbol{y}) = \mathbb{G}(\boldsymbol{x}, \boldsymbol{y})==\left(\mathbb{I}+\frac{\nabla_{\boldsymbol{x}} \nabla_{\boldsymbol{x}}}{k^{2}}\right) G(\boldsymbol{x}, \boldsymbol{y}),
    ```

    where ``k`` is a constant which depends on the electric permittivity, the
    magnetic permeability, and the angular frequency, and where

    ```math
    G(\boldsymbol{x},\boldsymbol{y}) = \frac{e^{ik|\boldsymbol{x}-\boldsymbol{y}|}}{4\pi |\boldsymbol{x}-\boldsymbol{y}|},
    ```

    is the Helmholtz Green's function.

    Defining for example a `MaxwellKernel` as done below, and calling the `assemble_ifgf` constructor, should work as expected. Note that, for performance reasons, tensor-valued kernels will typically return a `StaticArray`.

    ```@example maxwell-kernel

    using IFGF, StaticArrays, LinearAlgebra

    struct MaxwellKernel
        k::Float64
    end

    function (K::MaxwellKernel)(x,y)
        r   = x - y
        d   = norm(r)
        # helmholtz greens function
        g   = exp(im*K.k*d)/(4π*d)
        gp  = im*K.k*g - g/d
        gpp = im*K.k*gp - gp/d + g/d^2
        RRT = r*transpose(r) # rvec ⊗ rvecᵗ
        G   = g*LinearAlgebra.I + 1/K.k^2*(gp/d*LinearAlgebra.I + (gpp/d^2 - gp/d^3)*RRT)
        return (!iszero(d))*G
    end
    import IFGF: wavenumber
    wavenumber(K::MaxwellKernel) = K.k
    ```

## Advanced usage

The [`assemble_ifgf`](@ref) provides a high-level constructor for the `IFGFOp`
structure by making various default choices of clustering algorithms,
admisibility condition, how the interpolation cones should be created, etc. To
obtain a more granular control over these parameters, you have to create them
independently and pass them to the (defaulf) [`IFGF{T}`](@ref) constructor

Resuming the example for the Helmholtz kernel, we will go over all the steps to
manually construct the fields of the `IFGFOp`.

### Target tree

### Source tree

### Interaction list

### Cone domains

### `IFGFOp`
