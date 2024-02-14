# some simple point distributions for testing
using IFGF
using StaticArrays

function spheroid(N, r, center, zstretch)
    pts = Vector{IFGF.IFGF.Point3D}(undef, N)
    phi = π * (3 - sqrt(5)) # golden angle in radians
    for i in 1:N
        ytmp   = 1 - ((i - 1) / (N - 1)) * 2
        radius = sqrt(1 - ytmp^2)
        theta  = phi * i
        x      = cos(theta) * radius * r + center[1]
        y      = ytmp * r + center[2]
        z      = zstretch * sin(theta) * radius * r + center[3]
        pts[i] = SVector(x, y, z)
    end
    # Output
    return pts
end

sphere_uniform(N, r, center = (0, 0, 0)) = spheroid(N, r, center, 1)

function _cube(n, w, center = (0, 0, 0), order = 0)
    if order == 0
        cov = identity
    else
        P   = order
        v   = (x) -> (1 / P - 1 / 2) * ((1 - 2x))^3 + 1 / P * ((2x - 1)) + 1 / 2
        cov = (x) -> v(x)^P / (v(x)^P + v(1 - x)^P)
    end
    shift = IFGF.Point3D(center) .- w / 2
    n1d = ceil(Int, sqrt(n / 6))
    Δx = w / n1d
    x1d = [w * cov(x / w) for x in range(Δx / 2, w - Δx / 2, n1d)]
    pts = IFGF.Point3D[]
    for i in 1:n1d
        for j in 1:n1d
            # bottom
            x = SVector(x1d[i], x1d[j], 0) + shift
            push!(pts, x)
            # top
            x = SVector(x1d[i], x1d[j], w) + shift
            push!(pts, x)
            # left
            x = SVector(x1d[i], 0, x1d[j]) + shift
            push!(pts, x)
            # right
            x = SVector(x1d[i], w, x1d[j]) + shift
            push!(pts, x)
            # front
            x = SVector(0, x1d[i], x1d[j]) + shift
            push!(pts, x)
            # back
            x = SVector(w, x1d[i], x1d[j]) + shift
            push!(pts, x)
        end
    end
    return pts
end

cube_uniform(n, w, center = (0, 0, 0)) = _cube(n, w, center, 0)

cube_nonuniform(n, w, center = (0, 0, 0), order = 2) = _cube(n, w, center, order)
