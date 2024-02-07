"""
    abstract type AbstractSplitter

An `AbstractSplitter` is used to split a [`ClusterTree`](@ref). The interface
requires the following methods:
- `should_split(clt,splitter)` : return a `Bool` determining if the
  `ClusterTree` should be further divided
- `split!(clt,splitter)` : perform the splitting of the `ClusterTree` handling
  the necessary data sorting.

See [`GeometricSplitter`](@ref) for an example of an implementation.
"""
abstract type AbstractSplitter end

"""
    should_split(clt::ClusterTree, depth, splitter::AbstractSplitter)

Determine whether or not a `ClusterTree` should be further divided.
"""
function should_split(clt, depth, splitter::AbstractSplitter)
    return abstractmethod(splitter)
end

"""
    split!(clt::ClusterTree,splitter::AbstractSplitter)

Divide `clt` using the strategy implemented by `splitter`. This function is
reponsible of assigning the `children` and `parent` fields, as well as of
permuting the data of `clt`.
"""
function split!(clt, splitter::AbstractSplitter)
    return abstractmethod(splitter)
end

"""
    struct DyadicSplitter <: AbstractSplitter

Used to split an `N` dimensional `ClusterTree` into `2^N` children until at most
`nmax` points are contained in node.

## See also: [`AbstractSplitter`](@ref)
"""
@kwdef struct DyadicSplitter <: AbstractSplitter
    nmax::Int = 50
    keep_empty::Bool = false
end

function should_split(node::ClusterTree, depth, splitter::DyadicSplitter)
    return length(node) > splitter.nmax
end

function split!(parentcluster::ClusterTree, spl::DyadicSplitter)
    d = ambient_dimension(parentcluster)
    clusters = [parentcluster]
    rec = container(parentcluster)
    rec_center = center(rec)
    for i in 1:d
        pos = rec_center[i]
        nel = length(clusters) #2^(i-1)
        for _ in 1:nel
            clt = popfirst!(clusters)
            append!(clusters, _binary_split!(clt, i, pos; parentcluster))
        end
    end
    if !spl.keep_empty
        iempty = Int[]
        for (i, cluster) in enumerate(clusters)
            irange = index_range(cluster)
            isempty(irange) && (push!(iempty, i))
        end
        clusters = deleteat!(clusters, iempty)
    end
    parentcluster.children = clusters
    return parentcluster
end

"""
    struct DyadicMinimalSplitter <: AbstractSplitter

Similar to [`DiadicSplitter`](@ref), but the boundin boxes are shrank to the
minimal axis-aligned boxes at the end.

## See also: [`AbstractSplitter`](@ref)
"""
@kwdef struct DyadicMinimalSplitter <: AbstractSplitter
    nmax::Int = 50
    keep_empty::Bool = false
end

function should_split(node::ClusterTree, depth, splitter::DyadicMinimalSplitter)
    return length(node) > splitter.nmax
end

function split!(parentcluster::ClusterTree, spl::DyadicMinimalSplitter)
    # split as a dyadic splitter, then shrink bounding boxes
    split!(parentcluster, DyadicSplitter(spl.nmax, spl.keep_empty))
    # shrink bounding boxes to minimal one
    l2g = loc2glob(parentcluster)
    root_els = root_elements(parentcluster)
    clusters = children(parentcluster)
    for (i, cluster) in enumerate(clusters)
        irange = index_range(cluster)
        isempty(irange) && continue
        els = (root_els[l2g[j]] for j in irange)
        cluster.container = bounding_box(els, true)
    end
    return parentcluster
end

"""
    struct DyadicMaxDepthSplitter <: AbstractSplitter

Similar to [`DyadicSplitter`](@ref), but splits nodes until a maximum `depth` is reached.

## See also: [`AbstractSplitter`](@ref)
"""
@kwdef struct DyadicMaxDepthSplitter <: AbstractSplitter
    depth::Int
    keep_empty::Bool = false
end

function should_split(node::ClusterTree, depth, spl::DyadicMaxDepthSplitter)
    if !spl.keep_empty
        length(node) == 0 && (return false)
    end
    return depth < spl.depth
end

function split!(parentcluster::ClusterTree, spl::DyadicMaxDepthSplitter)
    # exactly like the dyadic splitter
    return split!(parentcluster, DyadicSplitter(-1, spl.keep_empty))
end
