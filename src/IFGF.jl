module IFGF

const PROJECT_ROOT =  pkgdir(IFGF)

using LinearAlgebra
using StaticArrays
using TimerOutputs
using Printf
using FFTW
using OrderedCollections

import AbstractTrees
import LinearAlgebra: mul!

import WavePropBase:
    HyperRectangle,
    ClusterTree,
    UniformCartesianMesh,
    center,
    CardinalitySplitter,
    DyadicSplitter,
    return_type,
    assert_concrete_type,
    partition_by_depth,
    ambient_dimension,
    container,
    radius,
    distance,
    isleaf,
    isroot,
    svector,
    children,
    parent,
    elements,
    low_corner,
    high_corner,
    ElementIterator,
    root_elements,
    index_range,
    coords,
    loc2glob,
    decrement_index,
    increment_index

include("utils.jl")
include("targettree.jl")
include("sourcetree.jl")
include("chebinterp.jl")
include("ifgfoperator.jl")

export
    # re-export WavePropBase
    WavePropBase,
    # types
    IFGFOp,
    ClusterTree,
    # methods
    assemble_ifgf,
    # macros
    @hprofile
end
