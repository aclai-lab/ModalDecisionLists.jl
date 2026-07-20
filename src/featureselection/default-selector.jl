using Random

"""
    DefaultFeatureSelector

A no-op feature selector that returns `cols` unchanged.

Use this selector when no feature sampling strategy is required and all
available features should be considered at every step.
"""
struct DefaultFeatureSelector <: FeatureSelector end

"""
    selectfeatures!(fs::DefaultFeatureSelector, cols, rng)

Return the complete feature list `cols` without modification.
"""
function selectfeatures!(
    fs::DefaultFeatureSelector, 
    cols::AbstractVector{Symbol}, 
    rng::AbstractRNG
)::Vector{Symbol}
    return cols
end