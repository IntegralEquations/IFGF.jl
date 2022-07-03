"""
    const TargetTree{N,T}

Type alias for a `ClusterTree` of points (represented as `SVector{N,T}`)
containing no additional data. See the documentation of
`ClusterTree` in [`WavePropBase`](https://github.com/WaveProp/WavePropBase.jl)
for more information.
"""
const TargetTree{N,T} = ClusterTree{Vector{SVector{N,T}},HyperRectangle{N,T},Nothing}

"""
    initialize_target_tree(;points,splitter)

Create the tree-structure for clustering the target `points` using the splitting
strategy of `splitter`. Returns a [`TargetTree`](@ref)
"""
function initialize_target_tree(;Xpoints,splitter)
    target_tree = ClusterTree(Xpoints,splitter)
    return target_tree
end
