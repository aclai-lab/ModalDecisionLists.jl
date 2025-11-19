module LossFunctions

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions


############################################################################################
############################# Loss Functions ###############################################

"""
    Calculates the laplace accuracy for the given data.\\
"""
function laplace_accuracy(
    y::AbstractVector{<:UInt32},
    w::AbstractVector=default_weights(length(y));
    nlabels::Integer,
    target_class::Union{Integer,Nothing} = nothing,
    kwargs...
)
    isempty(y) && return 0.0

    @assert length(w) == length(y) "weights and labels must have the same length"

    y_min = convert(Int64, minimum(y)) # this is necesssary, otherwise -y_min underflows when calculating y_offset

    y_offset = -y_min + 1

    # y_adj is just y normalized with a constant offset so that the smallest value is +1 
    y_adj = convert.(Int64, y) .+ y_offset

    # dist[i] is the number of occurrences of the label i in y_adj
    dist = counts(y_adj, nlabels, Weights(w))

    # number of effective labels (k) and number of matches for target class (target)
    k, target = nothing, nothing
    if !isnothing(target_class)
        # Converti target_class se era 0-based
        target_class_adj = target_class .+ y_offset

        # println("Number of pos elements in y: $(length(findall(label -> label==1, y)))")
        # println("Number of pos elements in y_adj: $(length(findall(label -> label==target_class_adj, y_adj)))")

        # println("y: $y")
        # println("y adj: $y_adj")

        # println("target class adj: $target_class_adj")
        # println("dist: $dist")
        k, target = 2, dist[Int(target_class_adj)]
    else
        k, target = nlabels, maximum(dist)
    end

    return 1 - ((target + 1) / (sum(dist) + k))
end

function entropy(
    y::AbstractVector{<:Integer},
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
    nlabels::Integer,
    kwargs...
)
    currdist = counts(ycurr, nlabels)
    prevdist = counts(yprev, nlabels)
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
