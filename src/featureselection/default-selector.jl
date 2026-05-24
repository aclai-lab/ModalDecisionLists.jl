using Random

struct DefaultFeatureSelector <: FeatureSelector end

function selectfeatures!(
    fs::DefaultFeatureSelector, 
    cols::AbstractVector{Symbol}, 
    rng::AbstractRNG
)::Vector{Symbol}
    return cols
end