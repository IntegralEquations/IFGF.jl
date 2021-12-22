module IFGF

const PROJECT_ROOT =  pkgdir(IFGF)

using LinearAlgebra
using StaticArrays
using TimerOutputs
using Printf
using FFTW

import LinearAlgebra: mul!

using WavePropBase
using WavePropBase.Utils
using WavePropBase.Geometry
using WavePropBase.Mesh
using WavePropBase.Interpolation
using WavePropBase.Trees

include("utils.jl")
include("targettree.jl")
include("sourcetree.jl")
include("chebinterp.jl")
include("ifgfoperator.jl")

export
    # re-export some modules from WavePropBase
    Geometry,
    Trees,
    Utils,
    # types
    IFGFOp,
    # macros
    @hprofile
end
