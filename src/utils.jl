"""
    share_interp_data(mode=true)

If set to `true`, neighboring cone interpolants will share their interpolation
data (which are, in this case, computed only once).
"""
function share_interp_data(mode = true)
    @eval _share_interp_data() = $mode
    mode
end
_share_interp_data() = true

"""
    use_fftw(mode=true)

Use the [FFTW](http://www.fftw.org) library for computing the Chebyshev transform.
"""
function use_fftw(mode=true)
    @eval _use_fftw() = $mode
    mode
end
_use_fftw()          = false

"""
    use_vectorized_chebeval(mode=true)

Use a vectorized version of the Chebyshev evaluation routine. Calling this
function will redefine `chebeval` to point to either `chebeval_vec` or
`chebeval_novec`, and thus may trigger code recompilation.
"""
function use_vectorized_chebeval(mode=true)
    if mode
        @eval chebeval(args...) = chebeval_vec(args...)
    else
        @eval chebeval(args...) = chebeval_novec(args...)
    end
    return nothing
end
# default choice
chebeval(args...) = chebeval_vec(args...)

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
function cone_domain_size_func(k, ds)
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

function cone_domain_size_func(k, ds::Number)
    if k == 0
        func = (node) -> begin
            N = ambient_dimension(node)
            ntuple(i->ds,N)
        end
    else
        # oscillatory case (e.g. Helmholtz, Maxwell)
        # k: wavenumber
        func = (node) -> begin
            N  = ambient_dimension(node)
            ds_tup = ntuple(i->ds,N)
            bbox = IFGF.container(node)
            w    = maximum(high_corner(bbox) - low_corner(bbox))
            δ    = max(k * w / 2,1)
            ds_tup ./ δ
        end
    end
    return func
end

"""
    modified_admissible_condition(target,source,[η])

A target and source are admissible under the *modified admissiblility condition*
(MAC) if the target box lies farther than `r*η` away, where `r` is the radius of
the source box and `η >= 1` is an adjustable parameter. By default, `η = N /
√N`, where `N` is the ambient dimension.
"""
function modified_admissibility_condition(target,source,η)
    # compute distance between source center and target box
    xc = source |> container |> center
    h  = source |> container |> radius
    bbox = container(target)
    dc   = distance(xc, bbox)
    # if target box is outside a sphere of radius h*η, consider it admissible.
    return dc > η*h
end

function modified_admissibility_condition(target,source)
    N = ambient_dimension(target)
    η = N / sqrt(N)
    modified_admissibility_condition(target,source,η)
end

"""
    _density_type_from_kernel_type(T)

Helper function to compute the expected density `V` type associated with a kernel type
`T`. For `T<:Number`, simply return `T`. For `T<:SMatrix`, return an `SVector`
with elements of the type `eltype(T)` and of length `size(T,2)` so that
multiplying an element of type `T` with an element of type `V` makes sense.
"""
function _density_type_from_kernel_type(T)
    if T <: Number
        return T
    elseif T <: Union{SMatrix,Adjoint}
        m,n = size(T)
        return SVector{n,eltype(T)}
    else
        error("kernel type $T not recognized")
    end
end

"""
    cheb_error_estimate(coefs::AbstractArray{T,N},dim)

Given an `N` dimensional array of coefficients , estimate the relative error
commited by the Chebyshev interpolant along dimension `dim`.
"""
function cheb_error_estimate(coefs::AbstractArray{T,N},dim) where {T,N}
    sz = size(coefs)
    I  = ntuple(N) do d
        if d == dim
            sz[d]:sz[d]
        else
            1:sz[d]
        end
    end
    norm(view(coefs,I...),2) / norm(coefs,2)
end

"""
    wavenumber(K::Function)

For oscillatory kernels, return the characteristic wavenumber (i.e `2π` divided
by he wavelength). For non-oscillatory kernels, return `0`.
"""
function wavenumber end



# @inline function selectlastdim(A::StaticArray,i)
#     N = ndims(A)
#     I = ntuple(N) do d
#         d == N ? i : Colon()
#     end
#     @inbounds getindex(A,I...)
#     # view(A,I...)
# end

# @inline function selectlastdim(A::SizedArray,i)
#     N = ndims(A)
#     I = ntuple(N) do d
#         d == N ? i : Colon()
#     end
#     # @inbounds getindex(A,I...)
#     view(A,I...)
# end

function reim_view(c::Array{Complex{T},N}) where {T,N}
    cs = reinterpret(T,c)
    n1 = size(cs,1)
    idxs_re = ntuple(i-> i==1 ? (1:2:n1) : Colon(),N)
    idxs_im = ntuple(i-> i==1 ? (2:2:n1) : Colon(),N)
    cr = view(cs,idxs_re...)
    ci = view(cs,idxs_im...)
    return cr,ci
end
