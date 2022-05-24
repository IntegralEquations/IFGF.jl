using IFGF
using Test
using Plots
using StaticArrays
plotlyjs()

n = 8
idxs = [SVector(1.0*i,1.0*j) for i in 0:n-1, j in 0:n-1] |> vec
p = IFGF.morton_perm(idxs)
# pidxs = idxs[p]
pidxs = sort(idxs;lt=IFGF.isless_zorder)
xx = [I[1] for I in pidxs]
yy = [I[2] for I in pidxs]
# plot(reverse(yy),xx)
plot(yy,xx,yflip=true)



n = 32
idxs = [(i,j,k) for i in 0:n-1, j in 0:n-1, k in 0:n-1] |> vec

sort!(idxs;lt=IFGF.isless_zorder)

xx = [I[1] for I in idxs]
yy = [I[2] for I in idxs]
zz = [I[3] for I in idxs]
plot(reverse(yy),xx,zz,lw=2)
plot(yy,xx,zz)


pts = rand(10)
IFGF.floats_to_uint(pts)
