const TargetTree{N,Td} = ClusterTree{N,Td,Nothing}

"""
    initialize_target_tree(;points,splitter)

Create the tree-structure for clustering the target `points` using the splitting
strategy of `splitter`. Returns a `TargetTree`.
"""
function initialize_target_tree(;points::Vector{SVector{N,T}},splitter) where {N,T}
    TargetTree{N,T}(points,splitter)
end
