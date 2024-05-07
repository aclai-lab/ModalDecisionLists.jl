module Measures

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase

# TODO è come se diventasse un problema biclasse ?
function laplace_accuracy(
    y::AbstractVector{<:Integer},
    w::AbstractVector=default_weights(length(y));
    target_class::Union{Integer,Nothing} = nothing,
    n_labels::Integer
)
    @show y
    tc = target_class
    dist = counts(y, n_labels)
    @show dist

    k, target = begin
        if !isnothing(tc)
            (2, dist[tc])
        else
            (length(dist), max(dist))
        end
    end
    return (-(target + 1) / (sum(dist) + k))
end

        # tc = rule.target_class
        # dist = rule.curr_class_dist
        # if tc is not None:
        #     k = 2
        #     target = dist[tc]
        # else:
        #     k = len(dist)
        #     target = bn.nanmax(dist)
        # return (target + 1) / (dist.sum() + k)

# TODO riguarda logica entropia !!! capire se è meglio versione bounded o unbounded
function entropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
    kwargs...
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    distribution = distribution[distribution .!= 0]
    # @show distribution
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    e = -sum(prob .* log2.(prob))
    return e
end

############################################################################################
#
# entropy_unbounded([0,1,2,3,4,5,6,7])
# 3.0
# julia> entropy_bounded([0,1,2,3,4,5,6,7])
# 1.0
# ==========================================

# | entropy_unbounded = entropy_bounded * log_2(n_classes)
# |        3.0        =       1.0       * log_2(8)
#

end # module
