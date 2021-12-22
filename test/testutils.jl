using ParametricSurfaces # for generating some simple point distributions
using LoopVectorization  # for automatically vectorizing some typical kernels

# some point distributions for testing
function cube_uniform_surface_mesh(dx)
    geo = ParametricSurfaces.Cube() # a cube with low corner at (0,0,0) and high corner (2,2,2)
    range = -1+dx/2:dx:1-dx/2
    pts   = Vector{Geometry.Point3D}()
    for ent in boundary(geo)
        for u in range, v in range
            push!(pts,ent((u,v)))
        end
    end
    return pts
end

function cube_nonuniform_surface_mesh(dx,p=2)
    geo = ParametricSurfaces.Cube() # a cube with low corner at (0,0,0) and high corner (2,2,2)
    range = -1+dx/2:dx:1-dx/2
    pts   = Vector{Geometry.Point3D}()
    # degenerate change of variables mapping [0,1] onto itself with p
    # derivatives vanishing at each endpoint
    cov   = Integration.KressP(order=p)
    # map cov to act on [-1,1]
    χ     = (u) -> begin
        û = 0.5*(u+1)
        ŝ = cov(û)
        s = 2*ŝ - 1
    end
    for ent in boundary(geo)
        for u in range, v in range
            push!(pts,ent((χ(u),χ(v))))
        end
    end
    return pts
end

function cube_uniform_volume_mesh(dx)
    range = dx/2:dx:1-dx/2
    [SVector(a,b,c) for a in range, b in range, c in range] |> vec
end

function sphere_uniform_surface_mesh(dx)
    geo   = ParametricSurfaces.Sphere(radius=1) # a sphere of radius 1 centered at (0,0,0)
    range = -1+dx/2:dx:1-dx/2
    pts   = Vector{Geometry.Point3D}()
    for ent in boundary(geo)
        for u in range, v in range
            push!(pts,ent((u,v)))
        end
    end
    return pts
end

# define some typical kernels
struct HelmholtzKernel
    k::Float64
end

function (K::HelmholtzKernel)(x,y)
    d = norm(x-y)
    v = exp(im*K.k*d)/(4*π*d)
    return (!iszero(d))*v
end

function IFGF.near_interaction!(C,K::HelmholtzKernel,X,Y,σ,I,J)
    Xm = reshape(reinterpret(Float64,X),3,:)
    Ym = reshape(reinterpret(Float64,Y),3,:)
    @views helmholtz3d_sl_vec!(C[I],Xm[:,I],Ym[:,J],σ[J],K.k)
end

function helmholtz3d_sl_vec!(C,X,Y,σ,k)
    m,n = size(X,2), size(Y,2)
    C_T = reinterpret(Float64, C)
    C_r = @views C_T[1:2:end,:]
    C_i = @views C_T[2:2:end,:]
    σ_T = reinterpret(Float64, σ)
    σ_r = @views σ_T[1:2:end,:]
    σ_i = @views σ_T[2:2:end,:]
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1,i] - Y[1,j])^2
            d2 += (X[2,i] - Y[2,j])^2
            d2 += (X[3,i] - Y[3,j])^2
            d  = sqrt(d2)
            s, c = sincos(k * d)
            zr = inv(4π*d) * c
            zi = inv(4π*d) * s
            C_r[i] += (!iszero(d))*(zr*σ_r[j] - zi*σ_i[j])
            C_i[i] += (!iszero(d))*(zi*σ_r[j] + zr*σ_i[j])
        end
    end
    return C
end

struct LaplaceKernel
end

function (K::LaplaceKernel)(x,y)
    d = norm(x-y)
    v = 1/(4*π*d)
    return (!iszero(d))*v
end

function IFGF.near_interaction!(C,K::LaplaceKernel,X,Y,σ,I,J)
    Xm = reshape(reinterpret(Float64,X),3,:)
    Ym = reshape(reinterpret(Float64,Y),3,:)
    @views laplace3d_sl_vec!(C[I],Xm[:,I],Ym[:,J],σ[J])
end

function laplace3d_sl_vec!(C,X,Y,σ)
    m,n = size(X,2), size(Y,2)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (X[1,i] - Y[1,j])^2
            d2 += (X[2,i] - Y[2,j])^2
            d2 += (X[3,i] - Y[3,j])^2
            # fast invsqrt code taken from here
            # https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-julia-8.html
            invd    = @fastmath Float64(1 / sqrt(Float32(d2)))
            invd = 1.5invd - 0.5d2 * invd * (invd * invd)
            C[i] += (!iszero(d2))*(inv(4π)*invd*σ[j])
            # C[i] += inv(4π*sqrt(d2))*σ[j] # significalty slower
        end
    end
    return C
end
