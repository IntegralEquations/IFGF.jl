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
Base.@kwdef struct DyadicSplitter <: AbstractSplitter
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
Base.@kwdef struct DyadicMinimalSplitter <: AbstractSplitter
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
        cluster.container = HyperRectangle(els, true)
    end
    return parentcluster
end

"""
    struct DyadicMaxDepthSplitter <: AbstractSplitter

Similar to [`DyadicSplitter`](@ref), but splits nodes until a maximum `depth` is reached.

## See also: [`AbstractSplitter`](@ref)
"""
Base.@kwdef struct DyadicMaxDepthSplitter <: AbstractSplitter
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

"""
    struct GeometricSplitter <: AbstractSplitter

Used to split a `ClusterTree` in half along the largest axis.
"""
Base.@kwdef struct GeometricSplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::GeometricSplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::GeometricSplitter)
    rec = cluster.container
    wmax, imax = findmax(high_corner(rec) - low_corner(rec))
    left_node, right_node = _binary_split!(cluster, imax, low_corner(rec)[imax] + wmax / 2)
    cluster.children = [left_node, right_node]
    return cluster
end

"""
    struct GeometricMinimalSplitter <: AbstractSplitter

Like [`GeometricSplitter`](@ref), but shrinks the children's containters.
"""
Base.@kwdef struct GeometricMinimalSplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::GeometricMinimalSplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::GeometricMinimalSplitter)
    rec = cluster.container
    wmax, imax = findmax(high_corner(rec) - low_corner(rec))
    mid = low_corner(rec)[imax] + wmax / 2
    predicate = (x) -> x[imax] < mid
    left_node, right_node = _binary_split!(cluster, predicate)
    cluster.children = [left_node, right_node]
    return cluster
end

"""
    struct PrincipalComponentSplitter <: AbstractSplitter
"""
Base.@kwdef struct PrincipalComponentSplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::PrincipalComponentSplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::PrincipalComponentSplitter)
    pts = cluster._elements
    irange = cluster.index_range
    xc = center_of_mass(cluster)
    # compute covariance matrix for principal direction
    l2g = loc2glob(cluster)
    cov = sum(irange) do i
        x = coords(pts[l2g[i]])
        return (x - xc) * transpose(x - xc)
    end
    v = eigvecs(cov)[:, end]
    predicate = (x) -> dot(x - xc, v) < 0
    left_node, right_node = _binary_split!(cluster, predicate)
    cluster.children = [left_node, right_node]
    return cluster
end

function center_of_mass(clt::ClusterTree)
    pts = clt._elements
    loc_idxs = clt.index_range
    l2g = loc2glob(clt)
    # w    = clt.weights
    n = length(loc_idxs)
    # M    = isempty(w) ? n : sum(i->w[i],glob_idxs)
    # xc   = isempty(w) ? sum(i->pts[i]/M,glob_idxs) : sum(i->w[i]*pts[i]/M,glob_idxs)
    M = n
    xc = sum(i -> coords(pts[l2g[i]]) / M, loc_idxs)
    return xc
end

"""
    struct CardinalitySplitter <: AbstractSplitter

Used to split a `ClusterTree` along the largest dimension if
`length(tree)>nmax`. The split is performed so the `data` is evenly distributed
amongst all children.

## See also: [`AbstractSplitter`](@ref)
"""
Base.@kwdef struct CardinalitySplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::CardinalitySplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::CardinalitySplitter)
    points = cluster._elements
    irange = cluster.index_range
    rec = container(cluster)
    _, imax = findmax(high_corner(rec) - low_corner(rec))
    l2g = loc2glob(cluster)
    med = median(coords(points[l2g[i]])[imax] for i in irange) # the median along largest axis `imax`
    predicate = (x) -> x[imax] < med
    left_node, right_node = _binary_split!(cluster, predicate)
    cluster.children = [left_node, right_node]
    return cluster
end
