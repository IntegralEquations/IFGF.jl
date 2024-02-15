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
    chebnodes(n,[domain])
    chebnodes(p::NTuple,[domain])
    chebnodes(::Val{P},[domain])

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
chebnodes((10,5),domain) # generate `10×5` Chebyshev nodes on the `[0,3]×[0,2]` rectangle.
```
"""
function chebnodes(n::Integer, ::Type{T}) where {T<:Real}
    return [T(cos((i - 1) * π / (n - 1))) for i in 1:n]
end

function chebnodes(p::NTuple, ::Type{T}) where {T}
    nodes1d = map(i -> chebnodes(i, T), p)
    iter = Iterators.product(nodes1d...)
    return iter
end

@generated function chebnodes(::Val{P}, ::Type{T}) where {P,T}
    nodes1d = map(i -> chebnodes(i, T), P)
    iter = Iterators.product(nodes1d...)
    nodes = SVector.(collect(iter))
    if prod(P) < 1e3
        nodes = SArray{Tuple{P...}}(nodes)
    end
    return :($nodes)
end

function chebnodes(p::Val{P}, rec::HyperRectangle{N,T}) where {P,N,T}
    refnodes = chebnodes(p, T)
    lc, hc = low_corner(rec), high_corner(rec)
    c0 = (lc + hc) / 2
    c1 = (hc - lc) / 2
    return map(x -> c0 + c1 .* x, refnodes)
end

function chebnodes(p::NTuple, rec::HyperRectangle)
    T = float_type(rec)
    refnodes = chebnodes(p, T)
    lc, hc = low_corner(rec), high_corner(rec)
    c0 = (lc + hc) / 2
    c1 = (hc - lc) / 2
    return map(x -> c0 + c1 .* x, refnodes)
end

"""
    chebtransform_fftw!(vals::AbstractArray[, plan])

Given a function values on the `chebnodes(p)`, where `p=size(vals)`, compute
the Chebyshev coefficients in-place. The underlying implementation uses the
`FFTW` library. A pre-computed `plan::FFTW.FFTWPlan` should be passed for
efficiency.
"""
function chebtransform_fftw!(vals::AbstractArray)
    plan = FFTW.plan_r2r!(copy(vals), FFTW.REDFT00)
    return chebtransform_fftw!(vals, plan)
end

function chebtransform_fftw!(vals::AbstractArray{<:Number,N}, plan::FFTW.FFTWPlan) where {N}
    FFTW.mul!(vals, plan, vals) # type-I DCT
    # renormalize the result to obtain the conventional
    # Chebyshev-polnomial coefficients
    s = size(vals)
    scale = 2^N / prod(n -> 2(n - 1), s)
    @. vals *= scale
    for dim in 1:N
        I = CartesianIndices(ntuple(i -> i == dim ? (1:1) : (1:s[i]), N))
        @views @. vals[I] /= 2
        I = CartesianIndices(ntuple(i -> i == dim ? (s[i]:s[i]) : 1:s[i], N))
        @views @. vals[I] /= 2
    end
    return vals
end

function chebtransform_fftw!(vals::AbstractArray{<:SVector{K}}, plan) where {K}
    coefs = ntuple(i -> chebtransform_fftw!([v[i] for v in vals], plan), Val{K}())
    return copyto!(vals, SVector{K}.(coefs...))
end

"""
    chebtransform_native!(vals)

Compute the Chebyshev coefficients given the values of a function on the
Chebyshev nodes of the second kind (i.e. the extrema of the Chebyshev
polynomials `Tₙ`). The  result is written in-place.
"""
function chebtransform_native!(vals, buff = copy(vals))
    D = ndims(vals)
    if D === 1
        # base case
        chebtransform1d!(vals, buff)
    else
        # recurse on N-1 dimensional slices
        inds_after = ntuple(Returns(:), D - 1)
        for i in 1:size(vals, 1)
            v = view(vals, i, inds_after...)
            b = view(buff, i, inds_after...)
            chebtransform_native!(v, b)
        end
        copy!(buff, vals)
        # finally do the first (contiguous) dimension
        vals1d = reshape(vals, size(vals, 1), :)
        buff1d = reshape(buff, size(vals, 1), :)
        for (v, b) in zip(eachcol(vals1d), eachcol(buff1d))
            chebtransform_native!(v, b)
        end
    end
    return vals
end

function chebtransform1d!(coefs, vals)
    N = length(vals)
    # DCT-I transform
    @inbounds for m in 0:N-1
        coefs[m+1] = 1 / 2 * (vals[1] + (-1)^m * vals[N])
        for n in 1:N-2
            coefs[m+1] += vals[n+1] * cos(π * m * n / (N - 1))
        end
        coefs[m+1] *= 2 / (N - 1)
    end
    # rescale coefs
    coefs[1] /= 2
    coefs[end] /= 2
    return coefs
end
chebtransform1d!(vals) = chebtransform1d!(vals, copy(vals))

"""
    chebeval(coefs,x,rec[,sz])

Evaluate the Chebyshev polynomial defined on `rec` with coefficients given by
`coefs` at the point `x`. If the size of `coefs` is known statically, its size
can be passed as a `Val` using the `sz` argument.
"""
@inline function chebeval(
    coefs::AbstractArray{T},
    x::SVector{N},
    rec::HyperRectangle,
    sz::Val{SZ},
) where {N,T,SZ}
    # translate to [-1,1]ᴺ
    # @assert x ∈ rec "$x ∉ $rec"
    x0 = @. (x - rec.low_corner) * 2 / (rec.high_corner - rec.low_corner) - 1
    # certain sizes get vectorized, while other (currently) don't
    if (SZ[1] ∈ (4, 8, 16, 32)) && (T == Float32 || T == Float64)
        x1, xr... = x0
        V = Vec{SZ[1],T}
        n = prod(SZ) ÷ SZ[1]
        coefs_ = reinterpret(V, coefs)
        f1d = _evaluate(xr, coefs_, Val{N - 1}(), 1, n, Val{SZ[2:end]}())
        return _evaluate(x1, f1d, Val{1}(), 1, SZ[1], Val(SZ[1:1]))
    elseif (SZ[1] ∈ (4, 8, 16, 32)) && (T == ComplexF32 || T == ComplexF64)
        x1, xr... = x0
        F = float_type(T)
        V = Vec{2 * SZ[1],F}
        n = prod(SZ) ÷ SZ[1]
        coefs_ = reinterpret(V, coefs)
        f1d = _evaluate(xr, coefs_, Val{N - 1}(), 1, n, Val{SZ[2:end]}())
        mask_real = ntuple(i -> 2(i - 1), Val{SZ[1]}()) |> Val
        mask_imag = ntuple(i -> 2(i - 1) + 1, Val{SZ[1]}()) |> Val
        f1d_real = shufflevector(f1d, mask_real)
        f1d_imag = shufflevector(f1d, mask_imag)
        vr = _evaluate(x1, f1d_real, Val{1}(), 1, SZ[1], Val(SZ[1:1]))
        vi = _evaluate(x1, f1d_imag, Val{1}(), 1, SZ[1], Val(SZ[1:1]))
        return Complex(vr, vi)
    else
        return _evaluate(x0, coefs, Val{N}(), 1, length(coefs), sz)
    end
end

#=
Adapted from `FastChebInterp` to focus on speed for small orders `p`. The coefs
in the IFGF algorithm is typically a view of some larger array which is reused
at different levels of the algorithn. We then pass the size of the coefficients
statically which allows for various improvements by the compiler (like loop unrolling).
=#
@inline @fastmath function _evaluate(x, c, ::Val{dim}, i1, len, sz::Val{SZ}) where {dim,SZ}
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
        for j in n-3:-1:1
            @inbounds bⱼ = muladd(2xd, bₖ, c[i1+j]) - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        return muladd(xd, bₖ, c₁) - bₖ₊₁
    else
        Δi = len ÷ n # column-major stride of current dimension

        # we recurse downward on dim for cache locality,
        # since earlier dimensions are contiguous
        dim′ = Val{dim - 1}()

        c₁ = _evaluate(x, c, dim′, i1, Δi, sz)
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁)
            c₂ = _evaluate(x, c, dim′, i1 + Δi, Δi, sz)
            return c₁ + xd * c₂
        end
        cₙ₋₁ = _evaluate(x, c, dim′, i1 + (n - 2) * Δi, Δi, sz)
        cₙ = _evaluate(x, c, dim′, i1 + (n - 1) * Δi, Δi, sz)
        bₖ = muladd(2xd, cₙ, cₙ₋₁)
        bₖ₊₁ = oftype(bₖ, cₙ)
        for j in n-3:-1:1
            cⱼ = _evaluate(x, c, dim′, i1 + j * Δi, Δi, sz)
            bⱼ = muladd(2xd, bₖ, cⱼ) - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        return muladd(xd, bₖ, c₁) - bₖ₊₁
    end
end

"""
    cheb_error_estimate(coefs::AbstractArray{T,N},dim)

Given an `N` dimensional array of coefficients , estimate the relative error
commited by the Chebyshev interpolant along dimension `dim`.
"""
function cheb_error_estimate(coefs::AbstractArray{T,N}, dim) where {T,N}
    sz = size(coefs)
    I = ntuple(N) do d
        if d == dim
            sz[d]:sz[d]
        else
            1:sz[d]
        end
    end
    return norm(view(coefs, I...), 2) / norm(coefs, 2)
end
