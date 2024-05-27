module Measures

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions

# TODO è come se diventasse un problema biclasse ?
function laplace_accuracy(
    y::AbstractVector{<:Integer},
    w::AbstractVector=default_weights(length(y));
    target_class::Union{Integer,Nothing} = nothing,
    n_labels::Integer,
    kwargs...
)
    N = length(y)
    distribution = counts(y, n_labels)
    target_class = isnothing(target_class) ?
            SoleModels.bestguess(y, suppress_parity_warning=true) : target_class
    k, target = begin
        if isnothing(target_class)
            (length(distribution), maximum(distribution))
        else
            (2, distribution[target_class])
        end
    end
    return 1 - (target + 1) / (N + k)
end

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


# ycurrent
# yprev
function significance_test(
    ycurr::AbstractVector{<:Integer},
    yprev::AbstractVector{<:Integer},
    alpha::Real;
    target_class::Union{Integer,Nothing} = nothing,
    n_labels::Integer,
    kwargs...
)
    currdist = counts(ycurr, n_labels)
    prevdist = counts(yprev, n_labels)
    if !isnothing(target_class)
        x = Vector{Real}([currdist[tc], sum(currdist) - currdist[tc]])
        y = Vector{Real}([prevdist[tc], sum(prevdist) - prevdist[tc]])
    else
        x = Vector{Real}(currdist)
        y = Vector{Real}(prevdist)
    end
    lrs = begin
        x[x .== 0] .= 1e-5
        y[y .== 0] .= 1e-5
        y = y * (sum(x)/sum(y))
        # Likelihood Ratio Statistic
        sum(x .* log.(x ./ y)) * 2
    end

    # Degrees of freedom
    df = length(currdist) - 1
    return ( lrs > 0 ) & (ccdf(Chisq(df), lrs) <= alpha)
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
