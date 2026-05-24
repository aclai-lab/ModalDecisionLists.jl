using DataFrames

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions
using CategoricalArrays: unwrap


"""
    count_labels_distribution(y::AbstractVector{<:Integer}, 
        nlabels::Integer, 
        w=default_weights(length(y));
        return_distribution::Bool = false
    )::Tuple{AbstractVector{<:Real}, Integer}


Compute the weighted distribution of labels in a vector.

# Arguments
- `y::AbstractVector{<:Integer}`: A vector of label indices.
- `nlabels::Integer`: The total number of distinct labels.
- `w`: Optional weight vector for each sample. Defaults to uniform weights if not provided.
- `return_distribution`: Optional flag to normalize the resulting distribution so that the sum of the elements is 1. 

# Returns
A tuple containing:
- `dist::AbstractVector{<:Real}`: A vector where `dist[i]` is the weighted count of occurrences of label `i`, with i going from 1 to nlabels.
- `y_offset::Integer`: The offset applied to normalize labels so the minimum label value becomes 1.

# Details
The function internally adjusts the label indices by offsetting them so that the minimum label value becomes 1.
This adjustment is necessary to properly index into the distribution vector. The offset value is returned
to allow for later denormalization if needed.
# Examples
```julia
# with 1-based labels and default weights
julia> y = UInt32[1, 2, 1, 3]
julia> counts, offset = count_labels_distribution(y, 5)  # 5 is the number of labels, although only {1, 2, 3} appear in y
([2, 1, 1, 0, 0], 0)  # offset is zero because the labels were already one-adjusted

# with non-1-based labels and custom weights
julia> y = UInt32[10, 12, 10, 11]
julia> w = [0.5, 1.0, 0.5, 2.0]
julia> counts, offset = count_labels_distribution(y, 12, w)
([1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], -9)
# offset = -9 so labels become 1, 3, 1, 2 after adjustment
# counts corresponds to weighted sums for adjusted labels 1 to 12
```
"""
function count_labels_distribution( 
    y::AbstractVector{<:Integer},
    n::Integer,
    w=default_weights(length(y));
    return_distribution::Bool = false
)::Tuple{AbstractVector{<:Real}, Integer}

    y_min = convert(Int64, minimum(y)) # this is necesssary, otherwise -y_min underflows when calculating y_offset
    y_offset = -y_min + 1

    # y_adj is just y normalized with a constant offset so that the smallest value is +1 
    y_adj = convert.(Int64, y) .+ y_offset

    # dist[i] is the number of occurrences of the label i in y_adj
    dist = counts(y_adj, n, Weights(w))

    return_distribution && (dist ./ sum(dist))

    return dist, y_offset   
end




"""
    calculate_prior_distribution(y::AbstractVector{<:CLabel})

Compute the empirical prior distribution of class labels in `y`.

# Arguments
- `y`: A vector of categorical labels.

# Returns
- `Dict{CLabel, Float64}`: relative frequency of each observed label in `y`.
"""
function calculate_prior_distribution(y::AbstractVector{<:CLabel})::Dict{CLabel, Real}
    counts = countmap(y)
    n = length(y)

    class_priors = Dict{CLabel, Float64}(
        label => get(counts, label, 0) / n
        # unwrap(label) => get(counts, label, 0) / n
        for (label, count) in counts
    )

    return class_priors
end




"""
maptointeger(y::AbstractVector{<:CLabel})

Map a categorical label vector to 1-based integer codes and return the ordered unique labels.

Arguments
- y: AbstractVector whose element type is a subtype of CLabel. Labels are compared using `==`.

Returns
- integer_y::Vector{UInt32}: a vector of length `length(y)` containing 1-based integer codes. Each distinct label v in `values` is assigned the code `i` where `values[i] == v`.
- values::Vector{eltype(y)}: the unique labels appearing in `y`, in the order of their first occurrence (as produced by `unique(y)`).

Notes
- The mapping is stable with respect to the first occurrence order of labels in `y`.
- The current implementation determines codes by comparing each label against the list of unique values (cost roughly O(n*m) where m = number of unique labels). For very large numbers of distinct labels a dictionary-based approach may be more efficient.

Example
```
julia> y = ["red","blue","red"]
julia> codes, vals = maptointeger(y)
 codes == UInt32[1,2,1]
 vals  == ["red","blue"]
```
"""
function maptointeger(y::AbstractVector{<:CLabel})
    # ordered values
    values = unique(y)
    integer_y = zeros(UInt32, length(y))

    for (i, v) in enumerate(values)
        integer_y[y.==v] .= i
    end
    return integer_y, values
end




"""
    preprocess_inputdata(X::AbstractDataFrame, y; remove_duplicate_rows=false)

Preprocess input data by optionally removing duplicate rows and handling missing values.

# Arguments
- `X::AbstractDataFrame`: A DataFrame containing the feature matrix.
- `y`: A vector of target labels corresponding to each row in `X`.
- `remove_duplicate_rows::Bool`: If `true`, removes duplicate rows from `X` and their 
  corresponding entries in `y`. If `false`, keeps all rows. Default: `false`.

# Returns
A tuple `(X_processed, y_processed)` containing:
- `X_processed::AbstractDataFrame`: The processed feature matrix with missing values removed.
- `y_processed`: The processed target vector with missing values removed.

# Details
- When `remove_duplicate_rows=true`: First checks if all rows in `X` are unique. If so, returns 
  the input unchanged. Otherwise, filters out duplicate rows and their corresponding labels.
- When `remove_duplicate_rows=false`: Keeps all rows as-is.
- In both cases, any rows with missing values in either `X` or `y` are removed.

# Example
```
julia> X = DataFrame(a=[1,2,1], b=[4,5,4])
julia> y = [0, 1, 0]
julia> X_clean, y_clean = preprocess_inputdata(X, y; remove_duplicate_rows=true)
```
"""
function preprocess_inputdata(
    X::AbstractDataFrame,
    y;
    remove_duplicate_rows=false
)
    if remove_duplicate_rows
        allunique(X) && return (X, y)
        nonunique_ind = nonunique(X)
        Xy = hcat(X[findall((!).(nonunique_ind)), :],
            y[findall((!).(nonunique_ind))]
        ) |> dropmissing
    else
        Xy = hcat(X[:, :], y[:]) |> dropmissing
    end
    return Xy[:, 1:(end-1)], Xy[:, end]
end



"""
    safe_reconstruct(obj, kwargs)

Reconstruct a model object using filtered keyword arguments.
This is essentially the same as `reconstruct` from Parameters.jl, but it does not throw an error if kwargs contains a field
that is not part of `obj`'s struct definition.
"""
function safe_reconstruct(obj, kwargs)
    # Set of valid field names for the object
    valid_keys = fieldnames(typeof(obj))

    # Filter the keys to only keep the ones that are actually parameters of the object's type before calling reconstruct
    to_apply = (; [k => v for (k, v) in kwargs if k in valid_keys]...)
    return reconstruct(obj; to_apply...)
end


function get_no_nil(a, b)
    return (isnothing(a)) ? b : a
end

""" returns an approximation of ln(n!) using Stirling's approximation for numerical stability and optimization """
log2_factorial(n::Integer)::Real = (n == 0) ? 0 : max(0, 0.5 * (1 + log2(π * n)) + n * log2(n / ℯ) + 0.1201753 / n)     # 0.1201753 / n is just a term that minimizes the approximation error without modifying the asymptotic relationship


""" returns an approximation of ln( n choose k ) using log2_factorial for numerical stability and optimization  """
log2binomial(n::Integer, k::Integer)::Real = (k == 0) ? 0 : log2_factorial(n) - log2_factorial(k) - log2_factorial(n - k)