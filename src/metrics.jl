module Metrics

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions
using ..ModalDecisionLists: count_labels_distribution

"""
    binary_accuracy(
        y::AbstractVector{<:String},
        y_pred::AbstractVector{<:String},
        target_class::String
    ) -> Float64

Computes binary accuracy for a target class by comparing true and predicted labels.

# Arguments
- `y::AbstractVector{<:String}`: Vector of true class labels.
- `y_pred::AbstractVector{<:String}`: Vector of predicted class labels.
- `target_class::String`: The specific class to evaluate accuracy for.

# Returns
- `Float64`: Accuracy value in the range [0, 1], where 1.0 indicates perfect accuracy.

# Details
This function treats the problem as binary classification by comparing each sample against the target class:
- A prediction is correct if both true and predicted labels match the target class, OR
- Both true and predicted labels differ from the target class (ex: predicted label is "other" and true label is not target_class).

This approach is useful when the model outputs some standard string like "other" for non-target classes.

# Throws
- `AssertionError`: If `y` and `y_pred` have different lengths.

# Examples
```julia
y = ["cat", "dog", "cat", "bird"]
y_pred = ["cat", "other", "cat", "cat"]
binary_accuracy(y, y_pred, "cat")  # Returns 0.75, because 3 predictions are right and one is wrong
```
"""
function binary_accuracy(
    y::AbstractVector{<:String},
    y_pred::AbstractVector{<:String},
    target_class::String
)

    @assert length(y_pred) == length(y) "The two vectors y and y_pred in accuracy should have the same length"

    total = length(y)
    correct = 0
    for i = 1 : total
        # since the models give a prediction "other" for another class, the condition is not simply y == y_pred 
        correct_bool = (y[i] == target_class && y_pred[i] == target_class) || (y[i] != target_class && y_pred[i] != target_class)
        correct += convert(Integer, correct_bool)
    end

    return correct / total
end


function gini_impurity(
    y::AbstractVector{<:UInt32},
    w::AbstractVector = default_weights(length(y));
    nlabels::Union{Integer, Nothing} = nothing,
    kwargs...
)
    isempty(y) && return Inf
    
    if isnothing(nlabels)
        nlabels = maximum(y)
    elseif nlabels <= 0
        throw(ArgumentError("`nlabels` must be ≥ 1, got $nlabels"))
    end

    dist, _ = count_labels_distribution(y, nlabels, Weights(w))
    filter!(!iszero, dist)
    length(dist) == 1 && return 0.0

    p = dist ./ sum(dist)
    return 1 - sum(abs2, p)  # abs2 is more efficient than .^2
end


function entropy(
    y::AbstractVector{<:UInt32},
    w::AbstractVector=default_weights(length(y));
    nlabels::Union{Integer, Nothing} = nothing,
    kwargs...
)
    isempty(y) && return Inf

    if isnothing(nlabels)
        nlabels = maximum(y)
    elseif nlabels <= 0
        throw(ArgumentError("`nlabels` must be ≥ 1, got $nlabels"))
    end

    # extract the labels' distribution, remove zero frequency elements and handle edge case in which all labels are in the same class
    distribution, _ = count_labels_distribution(y, nlabels, Weights(w))
    filter!(!iszero, distribution)
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

    dist, _ = count_labels_distribution(y, nlabels, w)

    # number of effective labels (k) and number of matches for target class (target)
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
        # Convert target_class if it's 0-based
        target_class_adj = target_class .+ y_offset

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
    currdist, _ = count_labels_distribution(ycurr, nlabels)
    prevdist, _ = count_labels_distribution(yprev, nlabels)
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
