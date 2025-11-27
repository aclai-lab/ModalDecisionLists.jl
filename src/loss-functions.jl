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
    Computes the (information) gain for a target class given the labels covered by a rule, their weights,
    the target class, the total number of positive samples in the original dataset (p) and the total number
    of negative samples in the original dataset (n)
"""
function gain(
    y::AbstractVector{<:UInt32},                # rules covered by an antecedent
    w::AbstractVector=default_weights(length(y));
    nlabels::Integer,
    target_class::Union{Integer,Nothing} = nothing,
    p::Integer = 0,
    n::Integer = 0,
    kwargs...
)
    tp = sum(y[y .== target_class]) # true positives
    fp = sum(y[y .!= target_class]) # true negatives

    
    """
    n = tn + fp
    -> tn = n - fp
    I want to calculate tn 
    """
    tn = n - fp

end




"""
    count_labels_distribution(y::AbstractVector{<:UInt32}, nlabels::Integer, w=default_weights(length(y)))::Tuple{AbstractVector{<:Real}, Integer}

Compute the weighted distribution of labels in a vector.

# Arguments
- `y::AbstractVector{<:UInt32}`: A vector of label indices.
- `nlabels::Integer`: The total number of distinct labels.
- `w`: Optional weight vector for each sample. Defaults to uniform weights if not provided.

# Returns
A tuple containing:
- `dist::AbstractVector{<:Real}`: A vector where `dist[i]` is the weighted count of occurrences of label `i`.
- `y_offset::Integer`: The offset applied to normalize labels so the minimum label value becomes 1.

# Details
The function internally adjusts the label indices by offsetting them so that the minimum label value becomes 1.
This adjustment is necessary to properly index into the distribution vector. The offset value is returned
to allow for later denormalization if needed.
"""
function count_labels_distribution(
    y::AbstractVector{<:UInt32},
    nlabels::Integer,
    w=default_weights(length(y))
)::Tuple{AbstractVector{<:Real}, Integer}
    y_min = convert(Int64, minimum(y)) # this is necesssary, otherwise -y_min underflows when calculating y_offset
    y_offset = -y_min + 1

    # y_adj is just y normalized with a constant offset so that the smallest value is +1 
    y_adj = convert.(Int64, y) .+ y_offset

    # dist[i] is the number of occurrences of the label i in y_adj
    dist = counts(y_adj, nlabels, Weights(w))

    return dist, y_offset   
end


"""
    laplace_accuracy(y::AbstractVector{<:UInt32}, w::AbstractVector=default_weights(length(y)); nlabels::Integer, target_class::Union{Integer,Nothing}=nothing, kwargs...) -> Float64

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
