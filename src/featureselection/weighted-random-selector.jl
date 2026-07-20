using StatsBase
using Random
using Tables

"""
    WeightedRandomFeatureSelector

A weighted random feature selector that chooses a fixed number of features at
each step using a softmax distribution over feature freshness.

The selector maintains a count for each feature that starts at `1` and is
updated on every call to `selectfeatures!`. Features that are selected have their
count reset to `1`; features that are not selected have their count incremented
by `1`. The selection logits are computed as `beta * count`, so features that
have not been chosen recently receive higher probability.

Arguments
- `features`: explicit feature names used to initialize the selector.
- `X`: a `PropositionalLogiset` from which feature names are inferred.
- `beta`: non-negative weight controlling the strength of the freshness bias.
  `beta == 0` gives uniform random selection.
- `num_per_step`: number of features selected on each call.
  If `num_per_step == length(cols)` or `num_per_step == -1`, all available
  features are returned unchanged.

This selector is stateful: repeated calls favor features that have not been
selected recently.
"""
mutable struct WeightedRandomFeatureSelector <: FeatureSelector
    feature_extraction_counts::Union{Nothing, Dict{Symbol, Integer}}
    beta::Real
    num_per_step::Integer       # number of features extracted at each step

    function WeightedRandomFeatureSelector(features::Vector{Symbol}, beta::Real, num_per_step::Integer)
        beta >= 0 || throw(ArgumentError("beta deve essere maggiore o uguale a 0 (ricevuto: $beta)"))

        counts = Dict{Symbol, Integer}(f => 1 for f in features)
        new(counts, beta, num_per_step)
    end

    function WeightedRandomFeatureSelector(X::PropositionalLogiset, beta::Real, num_per_step::Integer)
        beta >= 0 || throw(ArgumentError("beta deve essere maggiore o uguale a 0 (ricevuto: $beta)"))

        features = collect(Symbol, Tables.columnnames(Tables.columns(X)))
        counts = Dict{Symbol, Integer}(f => 1 for f in features)
    
        new(counts, beta, num_per_step)
    end
    
    function WeightedRandomFeatureSelector(beta::Real, num_per_step::Integer)
        beta >= 0 || throw(ArgumentError("beta deve essere maggiore o uguale a 0 (ricevuto: $beta)"))

        new(nothing, beta, num_per_step)
    end

end

"""
    selectfeatures!(fs::WeightedRandomFeatureSelector, cols, rng)

Select a weighted random subset of features from `cols` using the selector state.

The selector uses internal freshness counts to compute weights as a softmax over
`beta * count`. Selected features are reset to `1`, while unselected features
have their counts incremented by `1`.

Behavior
- If `num_per_step == length(cols)` or `num_per_step == -1`, returns all
  features unchanged.
- If `cols` differs from the previously tracked feature set, the internal counts
  are reinitialized to `1` for each feature in `cols`.
- Selection is performed without replacement according to the computed weights.

A larger `beta` increases bias toward features that were skipped previously.
"""
function selectfeatures!(
    fs::WeightedRandomFeatureSelector, 
    cols::AbstractVector{Symbol}, 
    rng::AbstractRNG
)::Vector{Symbol}    
    # dynamically initialize the features if they are not equal to 'cols' or not yet initialized
    if isnothing(fs.feature_extraction_counts) || !issetequal(keys(fs.feature_extraction_counts), cols)
        fs.feature_extraction_counts = Dict{Symbol, Integer}(f => 1 for f in cols)
    end

    # if we must return all the features anyways, or if the 'num_per_step' parameter is -1, then return all the features
    (fs.num_per_step == length(cols) || fs.num_per_step == -1) && return cols

    # extract the logits as β * counts
    logits = map(cols) do f
        return fs.beta * Float64(fs.feature_extraction_counts[f])
    end

    # apply the softmax on the logits. The subtraction of max_logit achieves greater numerical stability. 
    # This way, the largest weight will always be 1, and the ratio between each weight pair does not change 
    max_logit = maximum(logits)
    weights_vector = exp.(logits .- max_logit)

    # extract the features based on the weights
    selected = sample(rng, cols, Weights(weights_vector), fs.num_per_step, replace=false)

    selected_set = Set(selected)

    # update counts, so that feature_extraction_counts always contains the number of steps steps passed from the last time the feature was selected
    for f in keys(fs.feature_extraction_counts)
        if f in selected_set
            fs.feature_extraction_counts[f] = 1
        else
            fs.feature_extraction_counts[f] += 1
        end
    end

    return selected
end