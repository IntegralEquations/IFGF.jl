# for computing tensor products of chebyshev nodes of the second kind

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
