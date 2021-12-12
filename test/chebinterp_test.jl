using IFGF
using WavePropBase
using LinearAlgebra

# 1d cheb interp
p     = 5
n     = 32
sz    = Val((n,p))
coefs = rand(n*p)
x     = rand()
rec   = Interpolation.HyperRectangle(-1.,1.)

coefst = reshape(coefs,p,n) |> transpose |> collect |> vec

out = IFGF.chebeval1d(coefs,x,sz)
outvec = IFGF.chebeval1dvec(coefst,x,sz)
@info out-outvec

using BenchmarkTools
# @btime IFGF.chebeval1d($coefs,$x,$sz)

out = zeros(n)
@btime IFGF.chebeval1d!($out,$coefs,$x,$sz)
@btime IFGF.chebeval1dvec!($out,$coefst,$x,$sz)
