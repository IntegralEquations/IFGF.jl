module Laplace3D

using ..IFGF

"""
    lifgf3d(eps, sources; charges=nothing, dipvecs=nothing, targets=nothing,
    grad=false)

"""
function lifgf3d(target, source; charges = nothing, dipvecs = nothing, grad = false, tol)
    isnothing(charges) &&
        isnothing(dipvecs) &&
        error("Either charges or dipvecs must be provided")
    # dispatch to the appropriate kernel
    if !grad
        if isnothing(dipvecs) # single-layer
            K = SingleLayer
        elseif isnothing(charges) # double-layer
            K = GradSingleLayer
        else # combined-field
            error("combined field not yet implemented")
        end
    else
        error("gradient not yet implemented")
    end
    # assemble the matrix
    p = tol > 1e-4 ? (4, 4, 4) : tol > 1e-8 ? (8, 8, 8) : (16, 16, 16)
    L = assemble_ifgf(K, target, source; p)
    # create rhs with correct format
    if isnothing(dipvecs)
        rhs = charges
    elseif isnothing(charges)
        rhs = reinterpret(SVector{3,T}, dipvecs)
    else
        rhs = [vcat(c, d) for (c, d) in zip(charges, dipvecs)]
    end
    return L * rhs
end

function forward_map!(out, L::IFGFOp, charges = nothing, dipvecs = nothing)
    x = parse_rhs(charges, dipvecs)
    return mul!(out, L, x)
end

struct SingleLayer end

struct DoubleLayer end

function (K::SingleLayer)(x::SVector{}, y) end

end # module
