# some simple point distributions for testing
using IFGF
using StaticArrays

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
