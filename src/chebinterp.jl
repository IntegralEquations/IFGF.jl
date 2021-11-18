# this file contains some basic extensions to the FastChebInterp package
# https://github.com/stevengj/FastChebInterp.jl (MIT license). I am keeping them
# here for the moment, but at some point a PR may make sense. Because the IFGF
# package needs to evaluate Cheb polynomials of relatively low order, and
# because the order of the chebpoly the same for all nodes in the tree, it makes
# sense to:
# (a) pass the size statically so that evaluate can skip some conditionals and
# unroll loops. This yields significantly speed-ups in the evaluation of a
# chebpoly for given point
# (b) precompute a plan for the cheb transforms once and pass it to chebinterp.
# In fact the orders are small enough that the plan should just be a "direct
# summation" algorithm.
# (c) implement an in-place `chebinterp!` and `chebcoeffs!` which will use the
# array of values to store the chebyshev coefficiets
# (d) memoize the chebnodes in 1d
# (e) implement some product iterators for tensor product of cheb nodes

const CHEB2NODES1D = Dict{Int,Vector{Float64}}()

cheb2nodes(n::NTuple{N},lc::SVector{N},uc::SVector{N}) where {N} = map(x->SVector(x),cheb2nodes_iter(n,lc,uc))
function cheb2nodes_iter(n::NTuple{N},lc::SVector{N},uc::SVector{N}) where {N}
    nodes1d = ntuple(d->cheb2nodes(n[d],lc[d],uc[d]),N)
    iter = Iterators.product(nodes1d...)
    return iter
end
cheb2nodes_iter(n::NTuple,rec::HyperRectangle) = cheb2nodes(n,low_corner(rec),high_corner(rec))
function cheb2nodes(n::Integer,a::Number,b::Number)
    xcheb::Vector{Float64} = cheb2nodes(n)
    c0 = (a+b)/2
    c1 = (b-a)/2
    return @. c0 + c1*xcheb
end
function cheb2nodes(n)
    if !haskey(CHEB2NODES1D,n)
        nodes = [cos((i-1)*π /(n-1)) for i in 1:n]
        CHEB2NODES1D[n] = nodes
    end
    CHEB2NODES1D[n]
end

function chebinterp!(vals::AbstractArray{<:Any,N}, lb::SVector{N}, ub::SVector{N},plan::FFTW.FFTWPlan) where {N}
    Td    = promote_type(eltype(lb), eltype(ub))
    coefs = chebcoefs!(vals,plan)
    return ChebPoly{N,eltype(coefs),Td}(coefs, SVector{N,Td}(lb), SVector{N,Td}(ub))
end

function chebcoefs!(vals::AbstractArray{<:Number,N},plan::FFTW.FFTWPlan) where {N}
    coefs = FFTW.mul!(vals,plan,vals) # type-I DCT
    # renormalize the result to obtain the conventional
    # Chebyshev-polnomial coefficients
    s = size(coefs)
    # coefs ./= prod(n -> 2(n-1), s)
    scale = 2^N/prod(n -> 2(n-1), s)
    @. coefs *= scale
    for dim = 1:N
        I = CartesianIndices(ntuple(i -> i == dim ? (1:1) : (1:s[i]),N))
        @views @. coefs[I] /= 2
        I = CartesianIndices(ntuple(i -> i == dim ? (s[i]:s[i]) : 1:s[i],N))
        @views @. coefs[I] /= 2
    end
    return coefs
end

@fastmath function (interp::ChebPoly{N})(x::SVector{N,<:Real},sz::Val{SZ}) where {N,SZ}
    x0 = @. (x - interp.lb) * 2 / (interp.ub - interp.lb) - 1
    all(abs.(x0) .≤ 1 + 1e-8) || throw(ArgumentError("$x not in domain. lb = $(interp.lb), ub=$(interp.ub)"))
    return evaluate(x0, interp.coefs, Val{N}(), 1, length(interp.coefs),sz)
end

@fastmath function evaluate(x::SVector{N}, c::Array{<:Any,N}, ::Val{dim}, i1, len, sz::Val{SZ}) where {N,dim,SZ}
    n = SZ[dim]
    @inbounds xd = x[dim]
    if dim == 1
        c₁ = c[i1]
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁)
            return muladd(xd, c[i1+1], c₁)
        end
        @inbounds bₖ = muladd(2xd, c[i1+(n-1)], c[i1+(n-2)])
        @inbounds bₖ₊₁ = oftype(bₖ, c[i1+(n-1)])
        for j = n-3:-1:1
            @inbounds bⱼ = muladd(2xd, bₖ, c[i1+j]) - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        return muladd(xd, bₖ, c₁) - bₖ₊₁
    else
        Δi = len ÷ n # column-major stride of current dimension

        # we recurse downward on dim for cache locality,
        # since earlier dimensions are contiguous
        dim′ = Val{dim-1}()

        c₁ = evaluate(x, c, dim′, i1, Δi,sz)
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁)
            c₂ = evaluate(x, c, dim′, i1+Δi, Δi, sz)
            return c₁ + xd*c₂
        end
        cₙ₋₁ = evaluate(x, c, dim′, i1+(n-2)*Δi, Δi,sz)
        cₙ = evaluate(x, c, dim′, i1+(n-1)*Δi, Δi,sz)
        bₖ = muladd(2xd, cₙ, cₙ₋₁)
        bₖ₊₁ = oftype(bₖ, cₙ)
        for j = n-3:-1:1
            cⱼ = evaluate(x, c, dim′, i1+j*Δi, Δi,sz)
            bⱼ = muladd(2xd, bₖ, cⱼ) - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        return muladd(xd, bₖ, c₁) - bₖ₊₁
    end
end
