module IFGF

const PROJECT_ROOT = pkgdir(IFGF)

using LinearAlgebra
using StaticArrays
using TimerOutputs
using Printf
using FFTW
using OrderedCollections

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

export
    # types
    IFGFOp,
    ClusterTree,
    # methods
    assemble_ifgf,
    # macros
    @hprofile
end
