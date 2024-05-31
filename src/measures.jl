module Measures

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions


############################################################################################
############################# Loss Functions ###############################################

function laplace_accuracy(
    y::AbstractVector{<:Integer},
    w::AbstractVector=default_weights(length(y));
    n_labels::Integer,
    target_class::Union{Integer,Nothing} = nothing,
    kwargs...
)
    dist = counts(y, n_labels)

    k, target = begin
        if !isnothing(target_class)
            (2, dist[target_class])
        else
            (length(dist), maximum(dist))
        end
    end
    return 1 - ((target + 1) / (sum(dist) + k))
end

function entropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
    kwargs...
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    distribution = distribution[distribution .!= 0]

    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    e = -sum(prob .* log2.(prob))
    return e
end

############################################################################################
############################# Significance Test ############################################

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

end # module
