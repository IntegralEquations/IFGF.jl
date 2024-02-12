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

struct DoubleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct CombinedFieldKernel{Op} <: PDEKernel{Op}
    pde::Op
end

struct HessianSingleLayerKernel{Op} <: PDEKernel{Op}
    pde::Op
end

################################################################################
################################# HELMHOLTZ ####################################
################################################################################

"""
    struct Helmholtz{N, T}

Helmholtz equation in `N` dimension: Δu + k²u= 0. `k` is the wavenumber of
numeric type T.
"""
struct Helmholtz{N,T} <: AbstractPDE{N}
    k::T
end

function Base.show(io::IO, pde::Helmholtz)
    return print(io, "Δu + k²u = 0")
end

Helmholtz(; dim, k::T) where {T} = Helmholtz{dim,T}(k)

# TODO Is this correct?
function centered_factor(K::PDEKernel{Helmholtz{2,T}}, x, Y) where {T}
    yc = center(Y)
    d  = norm(x - yc)
    return hankelh1(0, K.pde.k * d)
end

function centered_factor(K::PDEKernel{Helmholtz{3,T}}, x, Y) where {T}
    yc = center(Y)
    d  = norm(x - yc)
    return exp(im*K.pde.k*d) / d
end

wavenumber(K::PDEKernel{<:Helmholtz}) = K.pde.k

function (SL::SingleLayerKernel{Helmholtz{N,T}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (hankelh1(0, SL.pde.k*d))
    elseif N == 3
        return filter * (exp(im*SL.pde.k*d) / d)
    else
        error("Not implemented for N = $N")
    end
end

function (SL::GradSingleLayerKernel{Helmholtz{N,T}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = SL.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 * k / d * hankelh1(1, k * d) * r)
    elseif N == 3
        return filter * (-1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * r)
    else
        error("Not implemented for N = $N")
    end
end

function (HS::HessianSingleLayerKernel{Helmholtz{N,T}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = HS.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        RRT = r * transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        val = (-1 * k^2 / d^2 * hankelh1(2, k * d) * RRT +
                                k / d * hankelh1(1, k * d) * I)
        return filter * val
    elseif N == 3
        ID = SMatrix{3,3,Complex{T},9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        RRT = r * transpose(r) # r ⊗ rᵗ
        term1 = 1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * ID
        term2 = RRT / d * exp(im * k * d) / d^4 *
                (3 * (d * im * k - 1) + d^2 * k^2)
        return filter * (term1 + term2)
    end
end


function (DL::DoubleLayerKernel{Helmholtz{N,T}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = DL.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (1 * k / d * hankelh1(1, k * d) * transpose(r))
    elseif N == 3
        return filter * (1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * transpose(r))
    else
        error("Not implemented for N = $N")
    end
end

function (CF::CombinedFieldKernel{Helmholtz{N,T}})(
    target::SVector{N,T},
    source::SVector{N,T},
    r = target - source,
) where {N,T}
    d = norm(r)
    k = CF.pde.k
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        dbllayer = (1 * k / d * hankelh1(1, k * d) * r)
        return filter * SVector{N + 1,Complex{T}}(hankelh1(0, k*d), dbllayer[1], dbllayer[2]) |> transpose
    elseif N == 3
        dbllayer = (1 / d^2 * exp(im * k * d) * (-im * k + 1 / d) * r)
        return filter * SVector{N + 1,Complex{T}}(exp(im * k * d) / d, dbllayer[1], dbllayer[2], dbllayer[3]) |>
               transpose
    else
        error("Not implemented for N = $N")
    end
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
    return 1 / d
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
        return filter * (1 / (d^2) * r)
    elseif N == 3
        return filter * (1 / (d^3) * r)
    else
        error("Not implemented for N = $N")
    end
end

function (::DoubleLayerKernel{Laplace{N}})(
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
        ID = SMatrix{3,3,T,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        RRT = r * transpose(r) # r ⊗ rᵗ
        return filter * (1 / (d^3) * ((ID - 3 * RRT / d^2)))
    end
end

################################################################################
# Implement a "simple" API for the IFGF operator when the kernels are
# implemented internally.

function forward_map!(out, L::IFGFOp; charges = nothing, dipvecs = nothing)
    K = L.kernel
    if K isa SingleLayerKernel
        @assert isnothing(dipvecs) "dipvecs should be nothing for single layer"
        @assert !isnothing(charges) "charges keyword required for single layer"
        mul!(out, L, charges)
    elseif K isa GradSingleLayerKernel
        @assert isnothing(dipvecs) "dipvecs should be nothing for adjoint double layer"
        @assert !isnothing(charges) "charges keyword required for adjoint double layer"
        mul!(out, L, charges)
    elseif K isa DoubleLayerKernel
        @assert isnothing(charges) "charges should be nothing for double layer"
        @assert !isnothing(dipvecs) "dipvecs keyword required for double layer"
        mul!(out, L, dipvecs)
    elseif K isa HessianSingleLayerKernel
        @assert isnothing(charges) "charges should be nothing for hessian of single layer"
        @assert !isnothing(dipvecs) "dipvecs keyword required for hessian of single layer"
        mul!(out, L, dipvecs)
    elseif K isa CombinedFieldKernel
        @assert !isnothing(charges) "charges are required for combined field"
        @assert !isnothing(dipvecs) "dipvecs are required for combined field"
        # concatenate charges and dipvecs
        x = [vcat(c, d) for (c, d) in zip(charges, dipvecs)]
        mul!(out, L, x)
    else
        error("argument / kernel mismatch")
    end
    return out
end

function forward_map(L::IFGFOp; charges = nothing, dipvecs = nothing)
    out = zeros(_output_type_forward_map(L, charges, dipvecs), size(L, 1))
    forward_map!(out, L; charges, dipvecs)
    return out
end

function _output_type_forward_map(L::IFGFOp, charges = nothing, dipvecs = nothing)
    K = L.kernel
    if K isa SingleLayerKernel
        @assert isnothing(dipvecs) "dipvecs should be nothing for single layer"
        @assert !isnothing(charges) "charges keyword required for single layer"
        return return_type(*, eltype(L), eltype(charges))
    elseif K isa GradSingleLayerKernel
        @assert isnothing(dipvecs) "dipvecs should be nothing for adjoint double layer"
        @assert !isnothing(charges) "charges keyword required for adjoint double layer"
        return return_type(*, eltype(L), eltype(charges))
    elseif K isa DoubleLayerKernel
        @assert isnothing(charges) "charges should be nothing for double layer"
        @assert !isnothing(dipvecs) "dipvecs keyword required for double layer"
        return return_type(*, eltype(L), eltype(dipvecs))
    elseif K isa HessianSingleLayerKernel
        @assert isnothing(charges) "charges should be nothing for hessian of single layer"
        @assert !isnothing(dipvecs) "dipvecs keyword required for hessian of single layer"
        return return_type(*, eltype(L), eltype(dipvecs))
    elseif K isa CombinedFieldKernel
        @assert !isnothing(charges) "charges are required for combined field"
        @assert !isnothing(dipvecs) "dipvecs are required for combined field"
        v = vcat(charges[1], dipvecs[1])
        return return_type(*, eltype(L), typeof(v))
    end
end

function plan_forward_map(
    pde,
    targets,
    sources;
    tol,
    grad = false,
    charges = false,
    dipvecs = false,
)
    !dipvecs && !charges && error("either charges or dipvecs should be provided")
    p = _tol_to_p(pde, tol)
    if !dipvecs && !grad
        @info "Planning single-layer operator for $pde"
        K = SingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    elseif !dipvecs && grad
        @info "Planning adjoint double-layer operator for $pde"
        K = GradSingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    elseif !charges && !grad
        @info "Planning double-layer operator for $pde"
        K = DoubleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    elseif dipvecs && !charges && grad
        @info "Planning hyper-singular operator for $pde"
        K = HessianSingleLayerKernel(pde)
        L = assemble_ifgf(K, targets, sources; p)
    else
        @info "Planning combined-field operator for $pde"
        @assert !grad "grad keyword incompatible with combined-field operator"
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

function _tol_to_p(::Helmholtz{3,T}, tol) where {T}
    tol > 1e-3 && return 4
    tol > 1e-6 && return 8
    return 16
end