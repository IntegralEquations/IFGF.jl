"""
    struct TargetTreeData

Struct used to store data associated with a node of the `TargetTree`. Currently
there is nothing there, but maybe one day there will be a downward pass in the
target tree, and it will need some fields to store data.
"""
struct TargetTreeData
end

"""
    const TargetTree{T,S,D}

Type alias for a `ClusterTree` with a data field `D` of type `TargetTreeData`.
"""
const TargetTree{T,S} = ClusterTree{T,S,<:TargetTreeData}
