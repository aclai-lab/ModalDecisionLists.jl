using ModalDecisionLists
using SoleModels: DecisionList, DecisionEnsemble
using SoleModels: default_weights
using SoleBase: CLabel
using SoleData: AbstractLogiset
using Tables

const RandomDecisionLists{O, A, W} = DecisionEnsemble{O, <:DecisionList, A, W} 

function RandomDecisionLists(
    lists::Vector{<:DecisionList}, 
    aggregation::Union{Nothing, Base.Callable},
    info::NamedTuple = (;),
)
    return DecisionEnsemble(
            lists, 
            aggregation, 
            nothing, # weights
            info        # TODO: merge with (type="RandomDecisionList")?
        )
end


lists(m::RandomDecisionLists) = models(m)
nlists(m::RandomDecisionLists) = length(lists(m))


"""
    build_rdl(X, y, poslabel, num_models, w=default_weights(length(y)); kwargs...)::DecisionEnsemble

Build a Random Decision List (RDL) ensemble classifier.

This function creates an ensemble of decision lists by training multiple individual decision lists
on random subsets of the data. Each decision list is trained using the `irepstar` algorithm on a
randomly sampled subset of instances and features.

# Arguments

- `X::AbstractLogiset`: The feature matrix (dataset)
- `y::AbstractVector{<:CLabel}`: The class labels
- `poslabel::CLabel`: The positive label (target class for rule learning)
- `num_models::Integer`: The number of decision lists to build in the ensemble
- `w::Union{Nothing, AbstractVector, Symbol}`: Instance weights (default: uniform weights)

# Keyword Arguments

- `use_bootstrapping::Bool`: If `true` (default), sample instances with replacement; 
  if `false`, sample without replacement
- `samples_ratio_per_model::Real`: Fraction of instances to use per model in range [0, 1] 
  (default: 1.0, meaning all instances)
- `n_subfeatures_per_model::Union{Integer, Nothing}`: Number of random features to use per model;
  if `nothing` (default), use all features
- `aggregation_function::Union{Nothing, Base.Callable}`: Optional function for ensemble prediction
  aggregation (default: `nothing`)
- `rand_seed::Union{Nothing, Integer}`: Random seed for reproducibility (default: `nothing`)
- `kwargs...`: Additional keyword arguments passed to the `irepstar` algorithm

# Returns

- `DecisionEnsemble`: A random decision list ensemble containing `num_models` individual 
  decision lists

# Raises

- `AssertionError`: If `samples_ratio_per_model` is not in [0, 1]
- `AssertionError`: If `num_models` ≤ 0
- `AssertionError`: If `n_subfeatures_per_model` is not in (0, num_features]

# Example

```julia
# Build an ensemble of 10 random decision lists
# using 80% of samples and all features per model
rdl = build_rdl(X_train, y_train, "positive_class", 10; samples_ratio_per_model=0.8)

# Make predictions
predictions = apply(rdl, X_test)
```
"""
function build_rdl(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    poslabel::CLabel,
    num_models::Integer,
    w::Union{Nothing, AbstractVector{U}, Symbol} = default_weights(length(y));
    
    use_bootstrapping::Bool = true,
    samples_ratio_per_model::Real = 1.0,
    n_subfeatures_per_model::Union{Integer, Nothing} = nothing,

    aggregation_function::Union{Nothing, Base.Callable} = nothing,
    rand_seed::Union{Nothing,Integer} = nothing,
    
    kwargs...
)::DecisionEnsemble where {U<:Real}
    @assert (0.0 ≤ samples_ratio_per_model ≤ 1.0) "Parameter `samples_ratio_per_model` must be in [0, 1]."
    @assert (num_models > 0) "Parameter `num_modelsc must be ≥ 1."
    
    num_features = nfeatures(X)
    
    !isnothing(n_subfeatures_per_model) && @assert (0 < n_subfeatures_per_model ≤ num_features) "Parameter `n_subfeatures_per_model` must be > 0 and ≤ than the number of features in the dataset."

    # Keep all features if the the user hasn't specified otherwise
    if isnothing(n_subfeatures_per_model)
        n_subfeatures_per_model = num_features
    end 

    num_samples = ninstances(X)
    all_feats = Tables.columnnames(Tables.columns(X))                   # list of feature names 
    n_samples_per_model = round(Integer, ninstances(X) * samples_ratio_per_model)

    models = DecisionList[]

    if !isnothing(rand_seed)
        Random.seed!(rand_seed)
    end

    # TODO: parallelization?
    for i = 1 : num_models
        # Extract 'n_samples_per_model' random integers in [1, num_samples] (with or without replacement depending on use_bootstrapping)
        if use_bootstrapping
            model_sample_indices = rand(1:num_samples, n_samples_per_model)    # this allows for sampling with replacement
        else
            permutated_indices = randperm(num_samples)
            model_sample_indices = permutated_indices[1:n_samples_per_model] 
        end

        # Extract 'n_subfeatures_per_model' features randomly 
        model_feature_indices = shuffle(all_feats)[1 : n_subfeatures_per_model]

        # use those indices to extract a dataset from X
        # X_model = slicedataset(X, model_sample_indices; return_view = true)        # select samples
        # X_model = Tables.subset(X_model, :, sub_feats)                              # select features
        X_model = X[model_sample_indices, model_feature_indices]            # select both features and indices


        y_model = @view y[model_sample_indices]
        w_model = (w isa AbstractVector) ? @view(w[model_sample_indices]) : w      # w might be nothing

        model = irepstar(X_model, y_model, poslabel, w_model; rand_seed=rand_seed, kwargs...) 
        push!(models, model)
    end

    return RandomDecisionLists(models, aggregation_function)    
end