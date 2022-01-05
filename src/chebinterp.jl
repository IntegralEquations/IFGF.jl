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

"""
    cheb2nodes(n,[domain])
    cheb2nodes(p::NTuple,[domain])
    cheb2nodes(::Val{P},[domain])

Return the `n` Chebyshev nodes of the second kind on the interval `[-1,1]`.

If passed a tuple `p`, return the tensor product of one-dimensinal nodes of size
`p`.

If passed a `Val{P}`, statically compute the tensor product nodes.

Finally, passing a `domain::HyperRectangle` shifts the interpolation nodes to
it.

```julia
lb = (0.,0.)
ub = (3.,2.)
domain = HyperRectangle(lb,ub)
cheb2nodes((10,5),domain) # generate `10×5` Chebyshev nodes on the `[0,3]×[0,2]` rectangle.
```
"""
function cheb2nodes(n)
    [cos((i-1)*π /(n-1)) for i in 1:n]
end

function cheb2nodes(p::NTuple)
    nodes1d = map(i->cheb2nodes(i),p)
    iter    = Iterators.product(nodes1d...)
    return iter
end

@generated function cheb2nodes(::Val{P}) where {P}
    nodes1d = map(i->cheb2nodes(i),P)
    iter    = Iterators.product(nodes1d...)
    nodes   = SVector.(collect(iter))
    if prod(P) < 1e3
        nodes  = SArray{Tuple{P...}}(nodes)
    end
    return :($nodes)
end

function cheb2nodes(p::Val{P},rec::HyperRectangle) where {P}
    refnodes = cheb2nodes(p)
    lc,hc    = low_corner(rec), high_corner(rec)
    c0       = (lc+hc)/2
    c1       = (hc-lc)/2
    map(x-> c0 + c1.*x,refnodes)
end

function cheb2nodes(p::NTuple,rec::HyperRectangle)
    refnodes = cheb2nodes(p)
    lc,hc    = low_corner(rec), high_corner(rec)
    c0       = (lc+hc)/2
    c1       = (hc-lc)/2
    map(x-> c0 + c1.*x,refnodes)
end

function chebcoefs!(vals::AbstractArray)
    plan = FFTW.plan_r2r!(copy(vals),FFTW.REDFT00)
    chebcoefs!(vals,plan)
end

function chebcoefs!(vals::AbstractArray{<:Number,N},plan::FFTW.FFTWPlan) where {N}
    FFTW.mul!(vals,plan,vals) # type-I DCT
    # renormalize the result to obtain the conventional
    # Chebyshev-polnomial coefficients
    s = size(vals)
    # coefs ./= prod(n -> 2(n-1), s)
    scale = 2^N/prod(n -> 2(n-1), s)
    @. vals *= scale
    for dim = 1:N
        I = CartesianIndices(ntuple(i -> i == dim ? (1:1) : (1:s[i]),N))
        @views @. vals[I] /= 2
        I = CartesianIndices(ntuple(i -> i == dim ? (s[i]:s[i]) : 1:s[i],N))
        @views @. vals[I] /= 2
    end
    return vals
end

function chebcoefs!(vals::AbstractArray{<:SVector{K}},plan) where {K}
    coefs = ntuple(i -> chebcoefs!([v[i] for v in vals],plan), Val{K}())
    copyto!(vals,SVector{K}.(coefs...))
end

@fastmath function chebeval(coefs,x::SVector{N,<:Real},rec::HyperRectangle,sz::Val{SZ}) where {N,SZ}
    x0 = @. (x - rec.low_corner) * 2 / (rec.high_corner - rec.low_corner) - 1
    return evaluate(x0, coefs, Val{N}(), 1, length(coefs),sz)
end

@fastmath function evaluate(x::SVector{N}, c, ::Val{dim}, i1, len, sz::Val{SZ}) where {N,dim,SZ}
    @inbounds n = SZ[dim]
    @inbounds xd = x[dim]
    if dim == 1
        @inbounds c₁ = c[i1]
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
