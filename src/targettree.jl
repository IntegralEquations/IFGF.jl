"""
    const TargetTree{N,T}

Type alias for a [`ClusterTree`](@ref) of points (represented as `SVector{N,T}`)
containing no additional data.
"""
const TargetTree{N,T} = ClusterTree{Vector{SVector{N,T}},HyperRectangle{N,T},Nothing}

"""
    initialize_target_tree(;points,splitter)

Create the tree-structure for clustering the target `points` using the splitting
strategy of `splitter`. Returns a [`TargetTree`](@ref)
"""
function initialize_target_tree(points, splitter)
    target_tree = ClusterTree(points, splitter)
    return target_tree
end
