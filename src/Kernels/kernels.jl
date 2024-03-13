"""
    const SAME_POINT_TOLERANCE = 1e-12

For singular kernels, two points are considered the same if their distance is
less than `SAME_POINT_TOLERANCE`.
"""
const SAME_POINT_TOLERANCE = 1e-12

"""
    abstract type AbstractPDE{N}

A partial differential equation in dimension `N`. `AbstractPDE` types are used
to define [`PDEKernel`](@ref)s.
"""
abstract type AbstractPDE{N} end

ambient_dimension(::AbstractPDE{N}) where {N} = N

"""
    abstract type PDEKernel{Op}

A kernel function associated a Green function (or derivatives of) a partial
differential equation.
"""
abstract type PDEKernel{Op} end

struct SingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct DoubleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct GradSingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct HessianSingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct CombinedFieldKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct GradCombinedFieldKernel{Op} <: PDEKernel{Op}
    pde::Op
end

@doc raw"""
    _plan_forward_map(pde, targets, sources, c, v, grad; kwargs...)

Assemble the `IFGFOp` for the forward map given by the sum

```math
u(\boldsymbol{x}_i) = \sum_{j=1}^N G(\boldsymbol{x}_i - \boldsymbol{y}_j) c_j + \gamma_{1,\boldsymbol{y}} G(\boldsymbol{x}_i - \boldsymbol{y}_j) v_j,
```

where $G$ is a Green's function of the PDE, $\gamma_{1,\boldsymbol{y}}$ its
generalized Neumann trace respect to the $\boldsymbol{y}$ variable, $c_j$
denotes the value of `x[j]`, and $v_j$ denotes the value of `v[j]`. If
`grad=true`, the gradient of the expression is computed instead. Either `c` or
`v` can be `nothing`, in which case the corresponding term is skipped in the
sum.

This function handles the logic needed to determine the appropriate kernel
function, as well values of `p` (number of interpolation points) and `h` (cone
meshsize), and then forwards the keyword arguments to [`assemble_ifgf`](@ref).
"""
function _plan_forward_map(pde::AbstractPDE, targets, sources, c, v, grad::Bool; kwargs...)
    # if either c or v is false, interpret that as not having that value (i.e. nothing)
    c == false && (c = nothing)
    v == false && (v = nothing)
    # if targets/sources are given as matrices, wrap them in a vector of points
    # structure
    target_equal_source = (targets === sources)
    N = ambient_dimension(pde)
    if targets isa Matrix
        @assert size(targets, 1) == N "targets should be a $N x m matrix"
        targets = _unsafe_wrap_vector_of_sarray(targets)
    end
    if sources isa Matrix
        if target_equal_source
            sources = targets
        else
            @assert size(sources, 1) == N "sources should be a $N x n matrix"
            sources = _unsafe_wrap_vector_of_sarray(sources)
        end
    end
    # figure out which kernel is needed
    if isnothing(v) && !grad
        @debug "Planning single-layer operator for $pde"
        K = SingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; kwargs...)
    elseif isnothing(v) && grad
        @debug "Planning adjoint double-layer operator for $pde"
        K = GradSingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; kwargs...)
    elseif isnothing(c) && !grad
        @debug "Planning double-layer operator for $pde"
        K = DoubleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; kwargs...)
    elseif isnothing(c) && grad
        @debug "Planning hyper-singular operator for $pde"
        K = HessianSingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; kwargs...)
    elseif !grad
        @debug "Planning combined-field operator for $pde"
        K = CombinedFieldKernel(pde)
        L = assemble_ifgf(K, targets, sources; kwargs...)
    else
        K = GradCombinedFieldKernel(pde)
        L = assemble_ifgf(K, targets, sources; kwargs...)
    end
    return L
end

"""
    _forward_map!(out, L, c, v, grad)

Execute the plan of [`_plan_forward_map`](@ref) and store the result in `out`.
"""
function _forward_map!(out, L::IFGFOp, c, v, grad, cname = "c", vname = "v")
    K = L.kernel
    if K isa SingleLayerKernel
        @assert isnothing(v) "$vname should be nothing for single layer"
        @assert !isnothing(c) "$cname keyword required for single layer"
        @assert !grad "grad keyword should be false for single layer"
        mul!(out, L, c)
    elseif K isa GradSingleLayerKernel
        @assert isnothing(v) "$vname should be nothing for adjoint double layer"
        @assert !isnothing(c) "$cname keyword required for adjoint double layer"
        @assert grad "grad keyword should be true for adjoint double layer"
        mul!(out, L, c)
    elseif K isa DoubleLayerKernel
        @assert isnothing(c) "$cname should be nothing for double layer"
        @assert !isnothing(v) "$vname keyword required for double layer"
        @assert !grad "grad keyword should be false for double layer"
        mul!(out, L, v)
    elseif K isa HessianSingleLayerKernel
        @assert isnothing(c) "$cname should be nothing for hessian of single layer"
        @assert !isnothing(v) "$vname keyword required for hessian of single layer"
        @assert grad "grad keyword should be true for hessian of single layer"
        mul!(out, L, v)
    elseif K isa CombinedFieldKernel
        @assert !isnothing(c) "$cname keyword required for combined field"
        @assert !isnothing(v) "$vname keyword required for combined field"
        @assert !grad "grad keyword should be false for combined field"
        # concatenate charges and dipvecs
        cv = [vcat(c, d) for (c, d) in zip(c, v)]
        mul!(out, L, cv)
    elseif K isa GradCombinedFieldKernel
        @assert !isnothing(c) "$cname keyword required for combined field"
        @assert !isnothing(v) "$vname keyword required for combined field"
        @assert grad "grad keyword should be true for combined field"
        # concatenate charges and dipvecs
        cv = [vcat(c, d) for (c, d) in zip(c, v)]
        mul!(out, L, cv)
    else
        error("argument / kernel mismatch")
    end
    return out
end

function _forward_map(L::IFGFOp, c, v, cname = "c", vname = "v")
    out = _allocate_output_forward_map(L, c, v)
    _forward_map!(out, L, c, v, cname, vname)
    return out
end

function _allocate_output_forward_map(L::IFGFOp, c, v)
    K = L.kernel
    T = if K isa SingleLayerKernel
        @assert isnothing(v) "dipvecs should be nothing for single layer"
        @assert !isnothing(c) "charges keyword required for single layer"
        return_type(*, eltype(L), eltype(c))
    elseif K isa GradSingleLayerKernel
        @assert isnothing(v) "dipvecs should be nothing for adjoint double layer"
        @assert !isnothing(c) "charges keyword required for adjoint double layer"
        return_type(*, eltype(L), eltype(c))
    elseif K isa DoubleLayerKernel
        @assert isnothing(c) "charges should be nothing for double layer"
        @assert !isnothing(v) "dipvecs keyword required for double layer"
        return_type(*, eltype(L), eltype(v))
    elseif K isa HessianSingleLayerKernel
        @assert isnothing(c) "charges should be nothing for hessian of single layer"
        @assert !isnothing(v) "dipvecs keyword required for hessian of single layer"
        return_type(*, eltype(L), eltype(v))
    elseif K isa CombinedFieldKernel
        @assert !isnothing(c) "charges are required for combined field"
        @assert !isnothing(v) "dipvecs are required for combined field"
        cv = vcat(c[1], v[1])
        return_type(*, eltype(L), typeof(cv))
    elseif K isa GradCombinedFieldKernel
        @assert !isnothing(c) "charges are required for combined field"
        @assert !isnothing(v) "dipvecs are required for combined field"
        cv = vcat(c[1], v[1])
        return_type(*, eltype(L), typeof(cv))
    else
        error("argument / kernel mismatch")
    end
    return zeros(T, size(L, 1))
end
