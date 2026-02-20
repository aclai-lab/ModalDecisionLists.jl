module Metrics

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions


# export gini_impurity
# export entropy
# export laplace_metric
# export laplace_accuracy
# export significance_test


function gini_impurity(
    y::AbstractVector{<:Integer},
    w::AbstractVector = default_weights(length(y))
)
    isempty(y) && return Inf
    
    dist = w isa Ones ? counts(y) : counts(y, Weights(w))
    filter!(!iszero, dist)
    length(dist) == 1 && return 0.0
    
    p = dist ./ sum(dist)
    return 1 - sum(abs2, p)  # abs2 è più efficiente di .^2
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


function laplace_metric(
    y::AbstractVector{<:UInt32},
    w::AbstractVector=default_weights(length(y));
    nlabels::Integer,
    kwargs...
)
    isempty(y) && return 0.0

    @assert length(w) == length(y) "weights and labels must have the same length"

    dist, y_offset = count_labels_distribution(y, nlabels, w)

    # number of effective labels (k) and number of matches for target class (target)
    # k, target = nothing, nothing
    k, target = nlabels, maximum(dist)

    return 1 - ((target + 1) / (sum(dist) + k))
end


"""
    laplace_accuracy(
        y::AbstractVector{<:UInt32}, 
        w::AbstractVector=default_weights(length(y)); 
        nlabels::Integer, 
        target_class::Union{Integer,Nothing}=nothing, kwargs...
    ) -> Float64

Computes a Laplace-accuracy based loss metric to estimate the coverage level of a rule.
A lower returned value indicates better coverage of target_class on y.

# Arguments
- `y::AbstractVector{<:UInt32}`: Vector of class labels (0-based or 1-based indexing).
- `w::AbstractVector`: Vector of weights for each sample. Defaults to uniform weights.
- `nlabels::Integer`: Total number of distinct labels in the dataset.
- `target_class::Union{Integer,Nothing}`: Specific class to evaluate. If `nothing`, uses the class with maximum count.
- `kwargs...`: Additional keyword arguments (unused).

# Returns
- `Float64`: A Laplace-smoothed accuracy value in the range [0, 1], where higher values indicate better accuracy.

# Details
The Laplace smoothing formula is: `1 - ((target + 1) / (sum(dist) + k))`
where:
- `target`: Count of samples matching the target class
- `sum(dist)`: Total count across all classes
- `k`: Number of effective labels (2 if `target_class` is specified, otherwise `nlabels`)

# Throws
- `AssertionError`: If the lengths of `w` and `y` do not match.
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

    dist, y_offset = count_labels_distribution(y, nlabels, w)

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




# ############################################################################################
# ############################# Significance Test ############################################

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


end # end of module
