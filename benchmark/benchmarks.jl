using BenchmarkTools
using IFGF
using StaticArrays

include(joinpath(IFGF.PROJECT_ROOT, "test", "simple_geometries.jl"))

const SUITE = BenchmarkGroup()

# point configuration used throughout the benchmarks
npts = 1_000_000
radius = 1
X = Y = sphere_uniform(npts, radius)

# parameters which are held constant across all benchmarks
tol = 1e-4

const PLAN = SUITE["plan"]

PLAN["Laplace3D"]["single layer"] = @benchmarkable(
    IFGF.plan_forward_map(IFGF.Laplace(; dim = 3), $X, $Y; tol = $tol, charges = true),
    evals = 1,
    samples = 1
)

PLAN["Helmholtz3D"]["single layer"] = @benchmarkable(
    IFGF.plan_forward_map(
        IFGF.Helmholtz(; dim = 3, k = 20π),
        $X,
        $Y;
        tol = $tol,
        charges = true,
    ),
    evals = 1,
    samples = 1
)

const FORWARD_MAP = SUITE["forward map"]

FORWARD_MAP["Laplace3D"]["single layer"] = @benchmarkable(
    IFGF.forward_map(L; charges = x),
    evals = 1,
    samples = 1,
    setup = (
        L = IFGF.plan_forward_map(
            IFGF.Laplace(; dim = 3),
            $X,
            $Y;
            tol = $tol,
            charges = true,
        );
        x = randn(Float64, npts)
    ),
)

FORWARD_MAP["Helmholtz3D"]["single layer"] = @benchmarkable(
    IFGF.forward_map(L; charges = x),
    evals = 1,
    samples = 1,
    setup = (
        L = IFGF.plan_forward_map(
            IFGF.Helmholtz(; dim = 3, k = 20π),
            $X,
            $Y;
            tol = $tol,
            charges = true,
        );
        x = randn(ComplexF64, npts)
    ),
)
