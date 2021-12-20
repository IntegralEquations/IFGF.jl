module IFGF

const PROJECT_ROOT =  pkgdir(IFGF)

using LinearAlgebra
using StaticArrays
using TimerOutputs
using Printf
using FFTW
using LoopVectorization
using SIMD

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
    # types
    UniformCartesianMesh,
    DyadicSplitter,
    CardinalitySplitter,
    GeometricMinimalSplitter,
    GeometricSplitter,
    TargetTree,
    SourceTree,
    IFGFOp,
    # methods
    initialize_target_tree,
    initialize_source_tree,
    compute_interaction_list!,
    compute_cone_list!,
    clear_interpolants!,
    cone_domain_size_func,
    # macros
    @hprofile
    # modules
    Utils

end
