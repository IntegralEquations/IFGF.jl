module IFGF

using LinearAlgebra
using StaticArrays
using TimerOutputs
using Printf
using RecipesBase
import AbstractTrees

using WavePropBase
using WavePropBase.Utils
using WavePropBase.Geometry
using WavePropBase.Mesh
using WavePropBase.Interpolation

WavePropBase.@import_interface

include("utils.jl")
include("targettree.jl")
include("sourcetree.jl")
include("ifgfoperator.jl")

export
    # types
    UniformCartesianMesh,
    DyadicSplitter,
    TargetTree,
    SourceTree,
    IFGFOperator,
    # methods
    initialize_target_tree,
    initialize_source_tree,
    compute_interaction_list!,
    compute_cone_list!,
    clear_interpolants!

WavePropBase.@export_interface

end
