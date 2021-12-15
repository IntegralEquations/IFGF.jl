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
# (d) statically compute the reference tensor-product chebnodes given the order

function cheb2nodes(n)
    [cos((i-1)*π /(n-1)) for i in 1:n]
end

@generated function cheb2nodes(::Val{P}) where {P}
    nodes1d = map(i->cheb2nodes(i),P)
    iter    = Iterators.product(nodes1d...)
    nodes   = SVector.(collect(iter))
    snodes  = SArray{Tuple{P...}}(nodes)
    return :($snodes)
end

function cheb2nodes(p::Val{P},rec::HyperRectangle) where {P}
    refnodes = cheb2nodes(p)
    lc,hc    = low_corner(rec), high_corner(rec)
    c0       = (lc+hc)/2
    c1       = (hc-lc)/2
    map(x-> c0 + c1.*x,refnodes)
end

function chebcoefs!(vals::AbstractArray{<:Number,N},plan::FFTW.FFTWPlan) where {N}
    # because vals may be a view with the wrong alignment, make a local copy of
    # it to use the precomputed plan, then copy it back to vals at the end
    coefs = copy(vals)
    FFTW.mul!(coefs,plan,coefs) # type-I DCT
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
    copyto!(vals,coefs)
    return vals
end

@fastmath function chebeval(coefs,x::SVector{N,<:Real},rec::HyperRectangle,sz::Val{SZ}) where {N,SZ}
    x0 = @. (x - rec.low_corner) * 2 / (rec.high_corner - rec.low_corner) - 1
    return evaluate(x0, coefs, Val{N}(), 1, length(coefs),sz)
end

function chebeval1d(coefs,x,p::Val{SZ}) where {SZ}
    n = SZ[1]
    P = SZ[2]
    kmax  = length(coefs) ÷ P
    out   = Vector{eltype(coefs)}(undef,kmax)
    chebeval1d!(out,coefs,x,p)
end

function chebeval1dvec(coefs,x,p::Val{SZ}) where {SZ}
    n = SZ[1]
    P = SZ[2]
    out   = Vector{eltype(coefs)}(undef,n)
    chebeval1dvec!(out,coefs,x,p)
end

using SIMD

function chebeval1dvec!(out,coefs,x,::Val{SZ}) where {SZ}
    # does a vectorized chebeval for `n` sets of cheb coefficients of length `P`
    # each. The coefs is supposed to be an `n × P` matrix.
    n = SZ[1]
    P = SZ[2]
    VECSIZE   = 4
    lane = VecRange{VECSIZE}(0)
    for k in 1:VECSIZE:n
        i1    = k + lane
        bₖ    = muladd(2x, coefs[i1+(P-1)*n], coefs[i1+(P-2)*n])
        bₖ₊₁  = oftype(bₖ, coefs[i1+(P-1)*n])
        for j = P-2:-1:2
            bⱼ = muladd(2x, bₖ, coefs[i1+(j-1)*n]) - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        out[k+lane] = muladd(x, bₖ, coefs[i1]) - bₖ₊₁
    end
    return out
end

function chebeval1d!(out,coefs,x,::Val{SZ}) where {SZ}
    n = SZ[1]
    P = SZ[2]
    for k in 1:n
        i1    = (k-1)*P+1
        bₖ    = muladd(2x, coefs[i1+P-1], coefs[i1+P-2])
        bₖ₊₁  = oftype(bₖ, coefs[i1+P-1])
        for j = P-2:-1:2
            bⱼ = muladd(2x, bₖ, coefs[i1-1+j]) - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        out[k] = muladd(x, bₖ, coefs[i1]) - bₖ₊₁
    end
    return out
end

@fastmath function evaluate(x::SVector{N}, c, ::Val{dim}, i1, len, sz::Val{SZ}) where {N,dim,SZ}
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

ChebPoly(rec::HyperRectangle,vals::Array) = ChebPoly(vals,low_corner(rec),high_corner(rec))
