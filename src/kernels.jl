const SAME_POINT_TOLERANCE = 1e-12

"""
    abstract type AbstractPDE{N}

A partial differential equation in dimension `N`. `AbstractPDE` types are used
to define `AbstractPDEKernel`s.
"""
abstract type AbstractPDE{N} end

ambient_dimension(::AbstractPDE{N}) where {N} = N

abstract type PDEKernel{Op} end

################################################################################
############################# Freespace green functions ########################
################################################################################

struct SingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct GradSingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct CombinedFieldKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct HessianSingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

################################################################################
################################# LAPLACE ######################################
################################################################################

"""
    struct Laplace{N}

Laplace equation in `N` dimension: Δu = 0.
"""
struct Laplace{N} <: AbstractPDE{N} end

function Base.show(io::IO, pde::Laplace)
    return print(io, "Δu = 0")
end

Laplace(; dim) = Laplace{dim}()

centered_factor(::PDEKernel{Laplace{2}}, x, Y) = 1

function centered_factor(::PDEKernel{Laplace{3}}, x, Y)
    yc = center(Y)
    d  = norm(x - yc)
    return 1/d
end

wavenumber(::PDEKernel{<:Laplace}) = 0

function (SL::SingleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (log(d))
    elseif N == 3
        return filter * (1 / d)
    else
        error("Not implemented for N = $N")
    end
end

function (::GradSingleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 / (d^2) * transpose(r))
    elseif N == 3
        return filter * (-1 / (d^3) * transpose(r))
    else
        error("Not implemented for N = $N")
    end
end

function (::CombinedFieldKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * SVector{N + 1,T}(log(d), -r[1] / d^2, -r[2] / d^2) |> transpose
    elseif N == 3
        return filter * SVector{N + 1,T}(1 / d, -r[1] / d^3, -r[2] / d^3, -r[3] / d^3) |>
               transpose
    else
        error("Not implemented for N = $N")
    end
end

function (::HessianSingleLayerKernel{Laplace{N}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (1 / (d^2) * ((I - 2 * r * transpose(r) / d^2)))
    elseif N == 3
        ID = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        RRT = r * transpose(r) # r ⊗ rᵗ
        return filter * (1 / (d^3) * ((ID - 3 * RRT / d^2)))
    end
end

################################################################################
# Implement a "simple" API for the IFGF operator when the kernels are
# implemented internally.

function forward_map!(out, L::IFGFOp; charges=nothing, dipvecs=nothing)
    K = L.kernel
    if K isa SingleLayerKernel
        @assert isnothing(dipvecs) "dipvecs should be nothing for single layer"
        @assert !isnothing(charges) "charges keyword required for single layer"
        mul!(out, L, charges)
    elseif K isa GradSingleLayerKernel
        @assert isnothing(charges) "charges should be nothing for grad single layer"
        @assert !isnothing(dipvecs) "dipvecs keyword required for grad single layer"
        mul!(out, L, dipvecs)
    elseif K isa CombinedFieldKernel
        @assert !isnothing(charges) "charges are required for combined field"
        @assert !isnothing(dipvecs) "dipvecs are required for combined field"
        # concatenate charges and dipvecs
        x = [vcat(c,d) for (c,d) in zip(charges, dipvecs)]
        mul!(out, L, x)
    end
    return out
end

function forward_map(L::IFGFOp; charges=nothing, dipvecs=nothing)
    out = zeros(_output_type_forward_map(L, charges, dipvecs), size(L,1))
    forward_map!(out, L; charges, dipvecs)
    return out
end

function _output_type_forward_map(L::IFGFOp, charges=nothing, dipvecs=nothing)
    K = L.kernel
    if K isa SingleLayerKernel
        @assert isnothing(dipvecs) "dipvecs should be nothing for single layer"
        @assert !isnothing(charges) "charges keyword required for single layer"
        return return_type(*, eltype(L), eltype(charges))
    elseif K isa GradSingleLayerKernel
        @assert isnothing(charges) "charges should be nothing for grad single layer"
        @assert !isnothing(dipvecs) "dipvecs keyword required for grad single layer"
        return return_type(*, eltype(L), eltype(dipvecs))
    elseif K isa CombinedFieldKernel
        @assert !isnothing(charges) "charges are required for combined field"
        @assert !isnothing(dipvecs) "dipvecs are required for combined field"
        v = vcat(charges[1], dipvecs[1])
        return return_type(*, eltype(L), typeof(v))
    end
end

function plan_forward_map(pde, targets, sources; tol, charges=nothing, dipvecs=nothing)
    isnothing(dipvecs) && isnothing(charges) && error("either charges or dipvecs should be provided")
    p = _tol_to_p(pde, tol)
    if isnothing(dipvecs)
        @info "Planning single-layer operafor for $pde"
        K = SingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    elseif isnothing(charges)
        @info "Planning double-layer operator for $pde"
        K = GradSingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    else
        @info "Planning combined-field operator for $pde"
        K = CombinedFieldKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    end
    return L
end

"""
    _tol_to_p(pde, tol::Real)

Heuristics to convert a tolerance `tol` to the polynomial order `p` for the IFGF
operator.
"""
function _tol_to_p(::Laplace{3}, tol)
    tol > 1e-3 && return 4
    tol > 1e-6 && return 8
    return 16
end
