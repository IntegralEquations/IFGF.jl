
struct IFGFOperator{T} <: AbstractMatrix{T}
    kernel
    target_tree::TargetTree
    source_tree::SourceTree
    p::NTuple
end

rowrange(ifgf::IFGFOperator)         = Trees.index_range(ifgf.target_tree)
colrange(ifgf::IFGFOperator)         = Trees.index_range(ifgf.source_tree)

kernel(op::IFGFOperator) = op.kernel
target_tree(op::IFGFOperator) = op.target_tree
source_tree(op::IFGFOperator) = op.source_tree
function Base.size(op::IFGFOperator)
    sizeX = op |> target_tree |> root_elements |> length
    sizeY = op |> source_tree |> root_elements |> length
    return (sizeX,sizeY)
end

function Base.getindex(op::IFGFOperator,i::Int,j::Int)
    error()
    # m,n = size(op)
    # ei = zeros()
    # getindex(op*ei,j)
end

"""
    IFGFOperator(kernel,
                 Ypoints,
                 Xpoints;
                 splitter,
                 p,
                 ds_func) where {V,U}

Create the source and target tree-structures for clustering the source `Ypoints`
and target `Xpoints` using the splitting strategy of `splitter`. Both trees are
initialized (i.e. the interaction list and the cone list are computed).

# Arguments
- `kernel`: kernel of the IFGF operator.
- `Ypoints`: vector of source points.
- `Xpoints`: vector of observation points.
- `splitter`: splitting strategy.
- `p`: the number of interpolation points
            per dimension in interpolation coordinates. In 3D, `p = (ns,nθ,nϕ)`.
            In 2D, `p = (ns,nθ)`.
- `ds_func`: function `ds_func(source_tree) = ds` that returns the size of the cone domains
             per dimension in interpolation coordinates. In 3D, `ds = (dss,dsθ,dsϕ)`.
             In 2D, `ds = (dss,dsθ)`.
"""
function IFGFOperator(kernel,
                      Ypoints,
                      Xpoints;
                      splitter,
                      p,
                      ds_func)
    # infer the type of the matrix
    U,V = eltype(Xpoints), eltype(Ypoints)
    T = hasmethod(return_type,Tuple{typeof(kernel)}) ? return_type(kernel) : Base.promote_op(kernel,U,V)
    assert_concrete_type(T)
    Tv = _density_type_from_kernel_type(T)
    source_tree = initialize_source_tree(;Ypoints,splitter,Xdatatype=typeof(Xpoints),Vdatatype=Tv)
    target_tree = initialize_target_tree(;Xpoints,splitter)
    compute_interaction_list!(source_tree,target_tree)
    compute_cone_list!(source_tree,target_tree,p,ds_func)
    ifgf = IFGFOperator{T}(kernel, target_tree, source_tree, p)
    return ifgf
end

function _density_type_from_kernel_type(T)
    if T <: Number
        return T
    elseif T <: SMatrix
        m,n = size(T)
        return SVector{n,eltype(T)}
    else
        error("kernel type $T not recognized")
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::IFGFOperator, x::AbstractVector, a::Number, b::Number;global_index=true,threads=false)
    # since the IFGF represents A = Pr*K*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end.
    T     = eltype(A)
    rtree = target_tree(A)
    ctree = source_tree(A)
    # permute input
    if global_index
        x         = x[ctree.loc2glob]
        y         = permute!(y,rtree.loc2glob)
        rmul!(x,a) # multiply in place since this is a new copy, so does not mutate exterior x
    else
        x = a*x # new copy of x
    end
    iszero(b) ? fill!(y,zero(eltype(y))) : rmul!(y,b)
    # extract the kernel and place a function barrier for heavy computation
    K     = kernel(A)
    if threads
        _gemv_threaded!(y,K,rtree,ctree,x,T,Val(A.p))
    else
        _gemv!(y,K,rtree,ctree,x,T,Val(A.p))
    end
    # permute output
    global_index && invpermute!(y,loc2glob(rtree))
    return y
end

function _gemv_threaded!(C,K,target_tree,source_tree,B,T,p::Val{P}) where {P}
    Tv = _density_type_from_kernel_type(T)
    # do some planning for the fft
    plan = FFTW.plan_r2r!(zeros(Tv,P),FFTW.REDFT00;flags=FFTW.PATIENT)
    partition = Trees.partition_by_depth(source_tree)
    nt = Threads.nthreads()
    Cthreads = [zero(C) for _ in 1:nt]
    dmax = length(partition)
    for d in dmax:-1:1
        Threads.@threads for node in partition[d]
            id = Threads.threadid()
            length(node) == 0 && continue
            _construct_chebyshev_interpolants_threads!(node,K,B,p,plan)
            _compute_far_interaction!(Cthreads[id],K,target_tree,node,p)
            _compute_near_interaction!(Cthreads[id],K,target_tree,source_tree,B,node)
        end
    end
    # reduction stage
    for i in 1:nt
        axpy!(1,Cthreads[i],C)
    end
    # since root has not parent, its data is not freed in the
    # `construct_chebyshev_interpolants`, so free it here
    free_buffer!(source_tree)
    return C
end

function _gemv!(C,K,target_tree,source_tree,B,T,p::Val{P}) where {P}
    Tv = _density_type_from_kernel_type(T)
    # do some planning for the fft
    plan = FFTW.plan_r2r!(zeros(Tv,P),FFTW.REDFT00;flags=FFTW.PATIENT)
    # the threading strategy is to partition the tree by depth, then go from
    # deepest to shallowest. Nodes of a fixed depth are mostly independent and
    # thus can spawn different tasks, (except for some "small" parts where they transfer
    # data to their parent or push into a dict). For that part, we use a lock.
    for node in PostOrderDFS(source_tree)
        length(node) == 0 && continue
        # allocate necessary interpolation data and perform necessary
        # precomputations
        @timeit_debug "Interpolant construction" _construct_chebyshev_interpolants!(node,K,B,p,plan)
        @timeit_debug "Far field contributions"  _compute_far_interaction!(C,K,target_tree,node,p)
        @timeit_debug "Near field contributions" _compute_near_interaction!(C,K,target_tree,source_tree,B,node)
    end
    # since root has not parent, its data is not freed in the
    # `construct_chebyshev_interpolants`, so free it here
    free_buffer!(source_tree)
end

function _construct_chebyshev_interpolants_threads!(node::SourceTree{N,Td,V,U,Tv},K,B,p::Val{P},plan) where {N,Td,V,U,Tv,P}
    Ypts = node |> root_elements
    allocate_buffer!(node,P)
    for (I,rec) in active_cone_idxs_and_domains(node)
        s          = cheb2nodes_iter(P,rec) # cheb points in parameter space
        x          = map(si->interp2cart(si,node),s) |> vec |> collect # cheb points in physical space
        irange = 1:length(x)
        jrange = index_range(node)
        vals       = getbuffer(node,I,P)
        if isleaf(node)
            near_interaction!(vals,K,x,Ypts,B,irange,jrange)
            for i in eachindex(x)
                xi = x[i] # cartesian coordinate of node
                vals[i] /= centered_factor(K,xi,node)
            end
        else
            for chd in children(node)
                length(chd) == 0 && continue
                for i in eachindex(x)
                    idxcone   = cone_index(x[i], chd)
                    si        = cart2interp(x[i],chd)
                    # transfer block's interpolant to parent's interpolants
                    rec  = cone_domains(chd)[idxcone]
                    coefs = getbuffer(chd,idxcone,P)
                    vals[i] += chebeval(coefs,si,rec,p) * transfer_factor(K,x[i],chd)
                end
            end
        end
        coefs = chebcoefs!(reshape(vals,P...),plan)
    end
    # interps on children no longer needed
    for chd in children(node)
        free_buffer!(chd)
    end
    return node
end

function _construct_chebyshev_interpolants!(node::SourceTree{N,Td,V,U,Tv},K,B,p::Val{P},plan) where {N,Td,V,U,Tv,P}
    Ypts = node |> root_elements
    allocate_buffer!(node,P)
    for (I,rec) in active_cone_idxs_and_domains(node)
        s          = cheb2nodes_iter(P,rec) # cheb points in parameter space
        x          = map(si->interp2cart(si,node),s) |> vec |> collect # cheb points in physical space
        irange = 1:length(x)
        jrange = index_range(node)
        vals       = getbuffer(node,I,P)
        if isleaf(node)
            @timeit_debug "leaf interpolants by direct sum" begin
                # leaf nodes compute their own interp vals
                near_interaction!(vals,K,x,Ypts,B,irange,jrange)
                for i in eachindex(x)
                    xi = x[i] # cartesian coordinate of node
                    vals[i] /= centered_factor(K,xi,node)
                end
            end
        else
            @timeit_debug "interpolant transfered from children" begin
                for chd in children(node)
                    length(chd) == 0 && continue
                    for i in eachindex(x)
                        idxcone   = cone_index(x[i], chd)
                        si        = cart2interp(x[i],chd)
                        # transfer block's interpolant to parent's interpolants
                        rec  = cone_domains(chd)[idxcone]
                        coefs = getbuffer(chd,idxcone,P)
                        vals[i] += chebeval(coefs,si,rec,p) * transfer_factor(K,x[i],chd)
                    end
                end
            end
        end
        @timeit_debug "compute cheb coeffs" begin
            coefs = chebcoefs!(reshape(vals,P...),plan)
        end
    end
    # interps on children no longer needed
    for chd in children(node)
        free_buffer!(chd)
    end
    return node
end

function cheb_error_estimate(coefs::Array{T,N}) where {T,N}
    sz = size(coefs)
    er = 0.0
    for I in CartesianIndices(coefs)
        any(Tuple(I) .== sz) || continue
        c = coefs[I]
        er = max(er,norm(c,2))
    end
    return er
end

# TODO: for each active cone, store a vector with (i,s) for each index in the
# far field that is going to be interpolated. This allows doing many of the
# computations offline at the cost of "a bit" extra memory, and more importantly
# allows for vectorization of the cheb poly evaluation
function _compute_far_interaction!(C,K,target_tree,node,p::Val{P}) where {P}
    Xpts  = target_tree |> root_elements
    for (I,idxs) in node.data.faridxs
        rec     = cone_domains(node)[I]
        coefs   = getbuffer(node,I,P)
        for i in idxs
            s    = cart2interp(coords(Xpts[i]),node)
            C[i]  += chebeval(coefs,s,rec,p) * centered_factor(K,Xpts[i],node)
        end
    end
    # for far_target in far_list(node)
    #     for i in index_range(far_target)
    #         x       = Xpts[i]
    #         I       = cone_index(coords(x),node)
    #         s       = cart2interp(coords(x),node)
    #         rec     = cone_domains(node)[I]
    #         coefs   = getbuffer(node,I,P)
    #         C[i]  += chebeval(coefs,s,rec,p) * centered_factor(K,Xpts[i],node)
    #     end
    # end
    return nothing
end

function _compute_near_interaction!(C,K,target_tree,source_tree,B,node)
    Xpts = target_tree |> root_elements
    Ypts = source_tree |> root_elements
    for near_target in near_list(node)
        # near targets use direct evaluation
        I = index_range(near_target)
        J = index_range(node)
        # FIXME: if we assume that if there is an i=j, then the points will be
        # the same and for singular kernels we should skip it. That is not
        # always the case (non-singular kernels and different surfaces). How
        # should we handle this?
        if !isempty(intersect(I,J))
            _near_interaction!(C,K,Xpts,Ypts,B,I,J)
        else
            near_interaction!(C,K,Xpts,Ypts,B,I,J)
        end
    end
    return nothing
end

function _compute_near_interaction_threads!(C,K,target_tree,source_tree,B,node)
    Xpts = target_tree |> root_elements
    Ypts = source_tree |> root_elements
    Threads.@threads for near_target in near_list(node)
        # near targets use direct evaluation
        I = index_range(near_target)
        J = index_range(node)
        near_interaction!(view(C,I),K,view(Xpts,I),view(Ypts,J),view(B,J))
    end
    return nothing
end

"""
    near_interaction!(C,K,X,Y,σ)

Compute `C[i] <-- C[i] + ∑ⱼ K(X[i],Y[j])*σ[j]`.
"""
near_interaction!(C,K,X,Y,σ,I,J) = _near_interaction!(C,K,X,Y,σ,I,J)

function _near_interaction!(C,K,X,Y,σ,I,J)
    for i in I
        for j in J
            i == j && continue
            C[i] += K(X[i], Y[j]) * σ[j]
        end
    end
end


# generic fallback. May need to be overloaded for specific kernels.
function centered_factor(K,x,ysource::SourceTree)
    yc = center(ysource)
    return K(x,yc)
end

# generic fallback. May need to be overloaded for specific kernels.
function transfer_factor(K,x,ysource::SourceTree)
    yparent = parent(ysource)
    return centered_factor(K,x,ysource)/centered_factor(K,x,yparent)
end

function Base.show(io::IO,ifgf::IFGFOperator)
    println(io,"IFGFOperator of $(eltype(ifgf)) with range $(rowrange(ifgf)) × $(colrange(ifgf))")
    ctree = source_tree(ifgf)
    nodes  = collect(PreOrderDFS(ctree))
    leaves = collect(Leaves(ctree))
    println("\t number of nodes:                $(length(nodes))")
    println("\t number of leaves:               $(length(leaves))")
    println("\t number of active cones:         $(sum(x->length(active_cone_idxs(x)),nodes))")
    println("\t maximum number of active cones: $(maximum(x->length(active_cone_idxs(x)),nodes))")
end
Base.show(io::IO,::MIME"text/plain",ifgf::IFGFOperator) = show(io,ifgf)
