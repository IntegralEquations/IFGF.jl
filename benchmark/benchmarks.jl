using BenchmarkTools
using IFGF
using StaticArrays

const SUITE = BenchmarkGroup()

# point configuration
npts = 1_000_000
X = Y = IFGF.points_on_unit_sphere(npts)

# parameters which are held constant across all benchmarks
p = 8
h = π/2

const PLAN = SUITE["plan"]

PLAN["Laplace3D"]["single layer"] = @benchmarkable(
    IFGF.laplace3d($X, $Y; charges, p = $p, h = $h),
    evals = 1,
    samples = 1,
    setup = (charges = randn(npts)),
)

PLAN["Helmholtz3D"]["single layer"] = @benchmarkable(
    IFGF.helmholtz3d(k, $X, $Y; charges, p = $p, h = $h),
    evals = 1,
    samples = 1,
    setup = (charges = randn(ComplexF64, npts); k = 20π),
)

PLAN["Stokes3D"]["single layer"] = @benchmarkable(
    IFGF.stokes3d($X, $Y; stoklet, p = $p, h = $h),
    evals = 1,
    samples = 1,
    setup = (stoklet = randn(3, npts)),
)
