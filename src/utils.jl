"""
    @hprofile

A macro which
- resets the default `TimerOutputs.get_defaulttimer` to zero
- execute the code block
- print the profiling details

This is useful as a coarse-grained profiling strategy to get a rough idea of
where time is spent. Note that this relies on `TimerOutputs` annotations
manually inserted in the code.
"""
macro hprofile(block)
    return quote
        TimerOutputs.enable_debug_timings(IFGF)
        reset_timer!()
        $(esc(block))
        print_timer()
    end
end

# double invsqrtQuake( double number )
#   {
#       double y = number;
#       double x2 = y * 0.5;
#       std::int64_t i = *(std::int64_t *) &y;
#       // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
#       i = 0x5fe6eb50c7b537a9 - (i >> 1);
#       y = *(double *) &i;
#       y = y * (1.5 - (x2 * y * y));   // 1st iteration
#       //      y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
#       return y;
#   }

# fast invsqrt code taken from here
# https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
@inline function invsqrt(x::Float64)
    y = @fastmath Float64(1 / sqrt(Float32(x)))
    # This is a Newton-Raphson iteration.
    return 1.5y - 0.5x * y * (y * y)
end

"""
    cone_domain_size_func(Δs₀::NTuple{N,T},k)

Returns an anonymous function `(node) -> Δs` that computes an appropriate size
`Δs` for the interpolation domain of `node` given an intial size `Δs₀` and a
wavenumber `k`. The function is constructed so as to scale `Δs₀` by the inverse
of the acoustic size of `node`.
"""
function cone_domain_size_func(k, ds = Float64.((1, π / 2, π / 2)))
    if k == 0
        func = (node) -> ds
    else
        # oscillatory case (e.g. Helmholtz, Maxwell)
        # k: wavenumber
        func = (node) -> begin
            bbox = IFGF.container(node)
            w    = maximum(high_corner(bbox) - low_corner(bbox))
            δ    = max(k * w / 2,1)
            ds ./ δ
        end
    end
    return func
end

"""
    modified_admissible_condition(target,source,[η])

A target and source are admissible under the *modified admissible condition*
(MAC) if the target box lies farther than `r*η` away, where `r` is the radius of
the source box and `η >= 1` is an adjustable parameter. By default, `η = N /
√N`, where `N` is the ambient dimension.
"""
function modified_admissible_condition(target,source,η)
    # compute distance between source center and target box
    xc = source |> container |> center
    h  = source |> container |> radius
    bbox = container(target)
    dc   = distance(xc, bbox)
    # if target box is outside a sphere of radius h*η, consider it admissible.
    return dc > η*h
end

function modified_admissible_condition(target,source)
    N = ambient_dimension(target)
    η = N / sqrt(N)
    modified_admissible_condition(target,source,η)
end

function _density_type_from_kernel_type(T)
    if T <: Number
        return T
    elseif T <: SMatrix
        m,n = size(T)
        return SVector{n,eltype(T)}
    else
        error("kernel type $T not recognized")
    end
end

function cheb_error_estimate(coefs::Array{T,N}) where {T,N}
    sz = size(coefs)
    er = 0.0
    for I in CartesianIndices(coefs)
        any(Tuple(I) .== sz) || continue
        c = coefs[I]
        er = max(er,norm(c,2))
    end
    return er
end
