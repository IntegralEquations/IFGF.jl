module IFGF

const PROJECT_ROOT = pkgdir(IFGF)

using FFTW
using LinearAlgebra
using OrderedCollections
using Printf
using StaticArrays
using Statistics: median
using TimerOutputs
using Bessels
using SIMD
using LoopVectorization

import AbstractTrees
import LinearAlgebra: mul!

include("utils.jl")
include("hyperrectangle.jl")
include("cartesianmesh.jl")
include("clustertree.jl")
include("splitter.jl")
include("targettree.jl")
include("sourcetree.jl")
include("chebinterp.jl")
include("ifgfoperator.jl")

# implementation of specific kernels
include("Kernels/kernels.jl")
include("Kernels/laplace.jl")
include("Kernels/helmholtz.jl")
include("Kernels/stokes.jl")

#
# include("simd.jl")

export
# methods
    assemble_ifgf
end
