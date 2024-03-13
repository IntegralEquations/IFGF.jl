```@meta
CurrentModule = IFGF
```

# [Custom kernels] (@id custom-kernel-section)

When defining your own kernels, there is one required as well as some optional
methods to overload in order to provide kernel-specific information to the
constructor. The table below summarizes the interface methods, where `Y` is a
`SourceTree` object, `x` is an `SVector` representing a point, and `K` is your
custom kernel:

| **Method's name**                          | **Required** | **Brief description**                                                                  |
| ------------------------------------------ | ------------ | -------------------------------------------------------------------------------------- |
| [`wavenumber(K)`](@ref)                    | Yes          | Return the wavenumber of your kernel                                                   |
| `return_type(K)` (@ref)                          | No           | Type of element returned by `K(x,y)`                                                   |
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
tree to compute the near interactions. Extending this function to make use of
e.g. SIMD instructions can significantly speed up some parts of the code.