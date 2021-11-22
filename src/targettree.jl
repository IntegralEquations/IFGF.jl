const TargetTree{N,Td,U} = ClusterTree{U,HyperRectangle{N,Td},Nothing}

"""
    initialize_target_tree(;points,splitter)

Create the tree-structure for clustering the target `points` using the splitting
strategy of `splitter`. Returns a `TargetTree`.
"""
function initialize_target_tree(;Xpoints::AbstractVector{U},splitter) where U
    target_tree = ClusterTree(Xpoints,splitter)
    return target_tree
end
