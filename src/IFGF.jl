module IFGF

using LinearAlgebra
using StaticArrays
using TimerOutputs
using Printf
using RecipesBase
using FastChebInterp
using FastChebInterp: ChebPoly
import AbstractTrees
using FFTW

using WavePropBase
using WavePropBase.Utils
using WavePropBase.Geometry
using WavePropBase.Mesh
using WavePropBase.Interpolation

WavePropBase.@import_interface

include("utils.jl")
include("chebinterp.jl")
include("targettree.jl")
include("sourcetree.jl")
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
    IFGFOperator,
    # methods
    initialize_target_tree,
    initialize_source_tree,
    compute_interaction_list!,
    compute_cone_list!,
    clear_interpolants!,
    # macros
    @hprofile
    # modules
    Utils

WavePropBase.@export_interface

end
