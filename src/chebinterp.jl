const CHEB2NODES1D = Dict{Int,Vector{Float64}}()

cheb2nodes(n::NTuple{N},lc::SVector{N},uc::SVector{N}) where {N} = map(x->SVector(x),cheb2nodes_iter(n,lc,uc))
function cheb2nodes_iter(n::NTuple{N},lc::SVector{N},uc::SVector{N}) where {N}
    nodes1d = ntuple(d->cheb2nodes(n[d],lc[d],uc[d]),N)
    iter = Iterators.product(nodes1d...)
    return iter
end
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

function chebcoefs!(vals::AbstractArray{<:Number,N},plan::FFTW.FFTWPlan) where {N}
    coefs = mul!(vals,plan,vals) # type-I DCT
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

function chebfit!(vals::AbstractArray{<:Any,N}, lb::SVector{N}, ub::SVector{N},plan::FFTW.FFTWPlan) where {N}
    Td    = promote_type(eltype(lb), eltype(ub))
    coefs = chebcoefs!(vals,plan)
    return ChebPoly{N,eltype(coefs),Td}(coefs, SVector{N,Td}(lb), SVector{N,Td}(ub))
end
