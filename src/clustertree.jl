"""
    mutable struct ClusterTree{T,S,D}

Tree structure used to cluster elements of type `V = eltype(T)` into containers of type `S`.
The method `center(::V)::SVector` is required for the clustering algorithms. An
additional `data` field of type `D` can be associated with each node to store
node-specific information (it defaults to `D=Nothing`).

# Fields:
- `_elements::T` : vector containing the sorted elements.
- `container::S` : container for the elements in the current node.
- `index_range::UnitRange{Int}` : indices of elements contained in the current node.
- `loc2glob::Vector{Int}` : permutation from the local indexing system to the
  original (global) indexing system used as input in the construction of the
  tree.
- `children::Vector{ClusterTree{N,T,D}}`
- `parent::ClusterTree{N,T,D}`
- `data::D` : generic data field of type `D`.
"""
mutable struct ClusterTree{T,S,D}
    _elements::T
    container::S
    index_range::UnitRange{Int}
    loc2glob::Vector{Int}
    buffer::Vector{Int}
    children::Vector{ClusterTree{T,S,D}}
    parent::ClusterTree{T,S,D}
    data::D
    # inner constructors handling missing fields.
    function ClusterTree{D}(
        els::T,
        container::S,
        loc_idxs,
        loc2glob,
        buffer,
        children,
        parent,
        data = nothing,
    ) where {T,S,D}
        clt = new{T,S,D}(els, container, loc_idxs, loc2glob, buffer)
        clt.children = isnothing(children) ? Vector{typeof(clt)}() : children
        clt.parent = isnothing(parent) ? clt : parent
        clt.data = isnothing(data) ? D() : data
        return clt
    end
end

# convenience functions and getters
"""
    root_elements(clt::ClusterTree)

The elements contained in the root of the tree to which `clt` belongs.
"""
root_elements(clt::ClusterTree) = clt._elements

"""
    index_range(clt::ClusterTree)

Indices of elements in `root_elements(clt)` which lie inside `clt`.
"""
index_range(clt::ClusterTree) = clt.index_range

children(clt::ClusterTree) = clt.children

Base.parent(clt::ClusterTree) = clt.parent

"""
    container(clt::ClusterTree)

Return the object enclosing all the elements of the `clt`.
"""
container(clt::ClusterTree) = clt.container

"""
    elements(clt::ClusterTree)

Iterable list of the elements inside `clt`.
"""
elements(clt::ClusterTree) = view(root_elements(clt), index_range(clt))

"""
    loc2glob(clt::ClusterTree)

The permutation from the (local) indexing system of the elements of the `clt` to
the (global) indexes used upon the construction of the tree.
"""
loc2glob(clt::ClusterTree) = clt.loc2glob

buffer(clt::ClusterTree) = clt.buffer

"""
    container_type(clt::ClusterTree)

Type used to enclose the elements of `clt`.
"""
container_type(::ClusterTree{T,S}) where {T,S} = S

"""
    element_type(clt::ClusterTree)

Type of elements sorted in `clt`.
"""
element_type(::ClusterTree{T}) where {T} = eltype(T)

isleaf(clt::ClusterTree) = isempty(clt.children)
isroot(clt::ClusterTree) = clt.parent == clt
hasdata(clt::ClusterTree) = isdefined(clt, :data)

diameter(node::ClusterTree) = diameter(container(node))
radius(node::ClusterTree) = radius(container(node))

"""
    distance(X::ClusterTree, Y::ClusterTree)

Distance between the containers of `X` and `Y`.
"""
function distance(node1::ClusterTree, node2::ClusterTree)
    return distance(container(node1), container(node2))
end

Base.length(node::ClusterTree) = length(index_range(node))

ambient_dimension(clt::ClusterTree) = ambient_dimension(container(clt))

"""
    ClusterTree(elements,splitter;[copy_elements=true])
    ClusterTree{D}(points,splitter;[copy_elements=true])

Construct a `ClusterTree` from the  given `elements` using the splitting
strategy encoded in `splitter`. If `copy_elements` is set to false, the
`elements` argument are directly stored in the `ClusterTree` and are permuted
during the tree construction.
"""
function ClusterTree{D}(
    elements,
    splitter = CardinalitySplitter();
    copy_elements = true,
    threads = false,
) where {D}
    copy_elements && (elements = deepcopy(elements))
    if splitter isa DyadicSplitter || splitter isa DyadicMaxDepthSplitter
        # make a cube for bounding box for quad/oct trees
        bbox = bounding_box(elements, true)
    else
        bbox = bounding_box(elements)
    end
    n = length(elements)
    irange = 1:n
    loc2glob = collect(irange)
    buffer = collect(irange)
    children = nothing
    parent = nothing
    #build the root, then recurse
    root = ClusterTree{D}(elements, bbox, irange, loc2glob, buffer, children, parent)
    _build_cluster_tree!(root, splitter, threads)
    # finally, permute the elements so as to use the local indexing
    copy!(elements, elements[loc2glob]) # faster than permute!
    return root
end
ClusterTree(args...; kwargs...) = ClusterTree{Nothing}(args...; kwargs...)

function _build_cluster_tree!(current_node, splitter, threads, depth = 0)
    if should_split(current_node, depth, splitter)
        split!(current_node, splitter)
        if threads
            Threads.@threads for child in children(current_node)
                _build_cluster_tree!(child, splitter, threads, depth + 1)
            end
        else
            for child in children(current_node)
                _build_cluster_tree!(child, splitter, threads, depth + 1)
            end
        end
    end
    return current_node
end

"""
    _binary_split!(cluster::ClusterTree,dir,pos;parentcluster=cluster)
    _binary_split!(cluster::ClusterTree,f;parentcluster=cluster)

Split a `ClusterTree` into two, sorting all elements in the process.
For each resulting child assign `child.parent=parentcluster`.

Passing a `dir` and `pos` arguments splits the `container` of `node`
along direction `dir` at position `pos`, then sorts all points into the
resulting  left/right nodes.

If passed a predicate `f`, each point is sorted
according to whether `f(x)` returns `true` (point sorted on
the left node) or `false` (point sorted on the right node). At the end a minimal
`HyperRectangle` containing all left/right points is created.
"""
function _binary_split!(cluster::ClusterTree, dir, pos; parentcluster = cluster)
    return _common_binary_split!(cluster, (dir, pos); parentcluster)
end
function _binary_split!(cluster::ClusterTree, f::Function; parentcluster = cluster)
    return _common_binary_split!(cluster, f; parentcluster)
end
function _common_binary_split!(
    cluster::ClusterTree{T,S,D},
    conditions;
    parentcluster,
) where {T,S,D}
    els = root_elements(cluster)
    l2g = loc2glob(cluster)
    buf = buffer(cluster)
    irange = index_range(cluster)
    # get split data
    npts_left, npts_right, left_rec, right_rec, buff =
        binary_split_data(cluster, conditions)
    @assert npts_left + npts_right == length(irange) "elements lost during split"
    copy!(view(l2g, irange), buff)
    # new ranges for cluster
    left_indices = (irange.start):((irange.start)+npts_left-1)
    right_indices = (irange.start+npts_left):(irange.stop)
    # create children
    clt1 = ClusterTree{D}(els, left_rec, left_indices, l2g, buf, nothing, parentcluster)
    clt2 = ClusterTree{D}(els, right_rec, right_indices, l2g, buf, nothing, parentcluster)
    return clt1, clt2
end
function binary_split_data(cluster::ClusterTree{T,S}, conditions::Function) where {T,S}
    f = conditions
    rec = container(cluster)
    els = root_elements(cluster)
    irange = index_range(cluster)
    n = length(irange)
    buff = view(cluster.buffer, irange)
    l2g = loc2glob(cluster)
    npts_left = 0
    npts_right = 0
    xl_left = xl_right = high_corner(rec)
    xu_left = xu_right = low_corner(rec)
    #sort the points into left and right rectangle
    for i in irange
        pt = center(els[l2g[i]])
        if f(pt)
            xl_left = min.(xl_left, pt)
            xu_left = max.(xu_left, pt)
            npts_left += 1
            buff[npts_left] = l2g[i]
        else
            xl_right = min.(xl_right, pt)
            xu_right = max.(xu_right, pt)
            buff[n-npts_right] = l2g[i]
            npts_right += 1
        end
    end
    # bounding boxes
    left_rec = S(xl_left, xu_left)
    right_rec = S(xl_right, xu_right)
    return npts_left, npts_right, left_rec, right_rec, buff
end
function binary_split_data(cluster::ClusterTree, conditions::Tuple{Integer,Real})
    dir, pos = conditions
    rec = container(cluster)
    els = root_elements(cluster)
    irange = index_range(cluster)
    n = length(irange)
    l2g = loc2glob(cluster)
    npts_left = 0
    npts_right = 0
    buff = view(cluster.buffer, irange)
    # bounding boxes
    left_rec, right_rec = split(rec, dir, pos)
    #sort the points into left and right rectangle
    for i in irange
        pt = center(els[l2g[i]])
        if pt in left_rec
            npts_left += 1
            buff[npts_left] = l2g[i]
        else # pt in right_rec
            # @assert pt in right_rec
            buff[n-npts_right] = l2g[i]
            npts_right += 1
        end
    end
    return npts_left, npts_right, left_rec, right_rec, buff
end

function Base.show(io::IO, tree::ClusterTree{T,S,D}) where {T,S,D}
    return print(io, "ClusterTree with $(length(tree.index_range)) elements of type $T")
end

function Base.summary(clt::ClusterTree)
    @printf "Cluster tree with %i elements" length(clt)
    nodes = collect(AbstractTrees.PreOrderDFS(clt))
    @printf "\n\t number of nodes: %i" length(nodes)
    leaves = collect(AbstractTrees.Leaves(clt))
    @printf "\n\t number of leaves: %i" length(leaves)
    points_per_leaf = map(length, leaves)
    @printf "\n\t min number of elements per leaf: %i" minimum(points_per_leaf)
    @printf "\n\t max number of elements per leaf: %i" maximum(points_per_leaf)
    depth_per_leaf = map(depth, leaves)
    @printf "\n\t min depth of leaves: %i" minimum(depth_per_leaf)
    @printf "\n\t max depth of leaves: %i" maximum(depth_per_leaf)
end

# interface to AbstractTrees. No children is determined by an empty tuple for
# AbstractTrees.
AbstractTrees.children(t::ClusterTree) = isleaf(t) ? () : t.children
AbstractTrees.nodetype(t::ClusterTree) = typeof(t)
