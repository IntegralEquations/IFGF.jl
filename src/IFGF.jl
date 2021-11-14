module IFGF

const PROJECT_ROOT =  pkgdir(IFGF)

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
using WavePropBase.Trees

WavePropBase.@import_interface

include("utils.jl")
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
    cone_domain_size_func,
    # macros
    @hprofile
    # modules
    Utils

WavePropBase.@export_interface

end
