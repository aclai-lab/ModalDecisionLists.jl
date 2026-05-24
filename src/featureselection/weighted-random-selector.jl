using StatsBase
using Random
using Tables

mutable struct WeightedRandomFeatureSelector <: FeatureSelector
    feature_extraction_counts::Dict{Symbol, Integer}
    alpha::Real
    num_per_step::Int       # number of features extracted at each step

    function WeightedRandomFeatureSelector(features::Vector{Symbol}, alpha::Real, num_per_step::Int)
        counts = Dict{Symbol, Integer}(f => 1 for f in features)

        new(counts, alpha, num_per_step)
    end

    function WeightedRandomFeatureSelector(X::PropositionalLogiset, alpha::Real, num_per_step::Int)
        features = collect(Symbol, Tables.columnnames(Tables.columns(X)))
        counts = Dict{Symbol, Integer}(f => 1 for f in features)
    
        new(counts, alpha, num_per_step)
    end
end

function selectfeatures!(
    fs::WeightedRandomFeatureSelector, 
    cols::AbstractVector{Symbol}, 
    rng::AbstractRNG
)::Vector{Symbol}    
    (fs.num_per_step == length(cols)) && return cols

    weights_vector = map(cols) do f
        return Float64(fs.feature_extraction_counts[f])^fs.alpha
    end

    selected = sample(rng, cols, Weights(weights_vector), fs.num_per_step, replace=false)

    selected_set = Set(selected)

    # update counts
    for f in keys(fs.feature_extraction_counts)
        if f in selected_set
            fs.feature_extraction_counts[f] = 1
        else
            fs.feature_extraction_counts[f] += 1
        end
    end

    return selected
end