using ModalDecisionLists
using SoleData
using SoleLogics
using SoleModels
using SoleModels: DecisionList, DecisionEnsemble
using SoleModels: default_weights
using SoleBase: CLabel
using SoleData: AbstractLogiset
using Tables

const SL = SoleLogics


struct RandomLists{O} <: SoleModels.AbstractDecisionEnsemble{O}
    models::Vector{DecisionList{O}}
    info::NamedTuple

    function RandomLists{O}(
        models::AbstractVector{<:DecisionList{O}},
        info::NamedTuple = (;)
    ) where {O}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        new{O}(collect(models), info)
    end

    function RandomLists(
        models::AbstractVector,
        info::NamedTuple = (;)
    )
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        O = Union{outcometype.(models)...}
        RandomLists{O}(models, info)
    end
    
end

models(m::RandomLists) = m.models
nmodels(m::RandomLists) = length(models(m))
isensemble(m::RandomLists) = true


"""
    apply_distribution(m::DecisionList, d::AbstractInterpretationSet; kwargs...)

Like `apply`, but instead of returning the predicted label for each instance,
returns a `Dict{CLabel, Float64}` with the m-estimate distribution of the
activated rule. Requires rules to have been built with `build_rule` (i.e. their
`info` must contain a `m_estimate` field).
"""
function apply_distribution(
    m::DecisionList,
    d::SL.AbstractInterpretationSet;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
)
    nsamp = ninstances(d)
    dists = Vector{Dict{CLabel, Float64}}(undef, nsamp)
    uncovered_idxs = 1:nsamp

    for rule in rulebase(m)
        length(uncovered_idxs) == 0 && break

        uncovered_d = slicedataset(d, uncovered_idxs; return_view = true)
        idxs_sat = findall(
            checkantecedent(rule, uncovered_d, check_args...; check_kwargs...)
        )
        idxs_sat = uncovered_idxs[idxs_sat]
        uncovered_idxs = setdiff(uncovered_idxs, idxs_sat)

        dist = info(rule).m_estimate   # Dict{CLabel, Real}
        foreach(i -> (dists[i] = dist), idxs_sat)
    end

    # fallback: default consequent for uncovered instances
    if length(uncovered_idxs) != 0
        default = defaultconsequent(m)
        default_dist = haskey(info(default), :m_estimate) ?
            info(default).m_estimate :
            Dict{CLabel, Float64}(outcome(default) => 1.0)
        foreach(i -> (dists[i] = default_dist), uncovered_idxs)
    end

    return dists
end



"""
    list_ensemble_aggregation(dists::AbstractVector{Dict{CLabel,Float64}}; kwargs...) -> CLabel

Aggregation function for an ensemble of DecisionLists using m-estimate distributions.
Each list contributes its activated rule's distribution, weighted by the probability
mass assigned to the predicted class (i.e. the Laplace-corrected precision of that rule).
The final prediction is the class with the highest total weighted score.

Intended to be passed as `aggregation` to `DecisionEnsemble`.
The outer weights (one per list) are handled by `DecisionEnsemble`'s `weighted_aggregation`.
"""
function list_ensemble_aggregation(
    dists::AbstractVector{Dict{CLabel,Float64}};
    kwargs...
)
    all_classes = union(keys.(dists)...)                            # union makes sure elements are not repeated
    scores = Dict{CLabel, Float64}(c => 0.0 for c in all_classes)

    for dist in dists
        for c in all_classes
            scores[c] += get(dist, c, 0.0)
        end
    end

    return argmax(scores)
end


function apply_rdl(
    m::RandomLists,
    d::SL.AbstractInterpretationSet;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
)
    nsamp = ninstances(d)

    # one Vector{Dict} per list, each with length equal to "nsamp"
    all_dists = [
        apply_distribution(list, d; check_args, check_kwargs)
        for list in models(m)
    ]

    return [
        list_ensemble_aggregation([all_dists[l][i] for l in eachindex(models(m))])
        for i in 1:nsamp
    ]
end


function apply_rdl(
    m::RandomLists,
    i::SL.AbstractInterpretation;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;)
)
    dists = [
        apply_distribution(list, i; check_args, check_kwargs)
        for list in models(m)
    ]
    return list_ensemble_aggregation(dists)
end






function build_random_lists(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    num_models::Integer,
    w::Union{Nothing, AbstractVector{U}, Symbol} = default_weights(length(y));

    featurenames::Union{Nothing,Vector{Symbol}}=nothing,
    
    use_bootstrapping::Bool = true,
    samples_ratio_per_model::Real = 1.0,
    n_subfeatures_per_model::Union{Integer, Nothing} = nothing,

    rng::AbstractRNG = Random.default_rng(),
    model_wrapper::Base.Callable = irepstar,
    
    kwargs...
)::RandomLists where {U<:Real}
    @assert (0.0 ≤ samples_ratio_per_model ≤ 1.0) "Parameter `samples_ratio_per_model` must be in [0, 1]."
    @assert (num_models > 0) "Parameter `num_models must be ≥ 1."
    
    num_features = nfeatures(X)
    
    !isnothing(n_subfeatures_per_model) && @assert (0 < n_subfeatures_per_model ≤ num_features) "Parameter `n_subfeatures_per_model` must be > 0 and ≤ than the number of features in the dataset."

    # Keep all features if the the user hasn't specified otherwise
    if isnothing(n_subfeatures_per_model)
        n_subfeatures_per_model = num_features
    end 

    class_priors = calculate_prior_distribution(y)
    
    num_samples = ninstances(X)
    all_feats = collect(Symbol, Tables.columnnames(Tables.columns(X)))                   # list of features
    n_samples_per_model = round(Integer, ninstances(X) * samples_ratio_per_model)

    models = Vector{DecisionList}(undef, num_models)

    for model_num = 1 : num_models
		local_rng = copy(rng)

        # Extract 'n_samples_per_model' random integers in [1, num_samples] (with or without replacement depending on use_bootstrapping)
        if use_bootstrapping
            model_sample_indices = rand(local_rng, 1:num_samples, n_samples_per_model)    # this allows for sampling with replacement
        else
            permutated_indices = randperm(local_rng, num_samples)
            model_sample_indices = permutated_indices[1:n_samples_per_model] 
        end

        # Extract 'n_subfeatures_per_model' features randomly 
		model_feature_names = if (n_subfeatures_per_model != num_features)
            shuffle(local_rng, all_feats)[1 : n_subfeatures_per_model]
        else
            all_feats
        end

        # use those indices to extract a dataset from X
        X_model = X[model_sample_indices, model_feature_names]            # select sampled features and instances
        y_model = y[model_sample_indices]
        w_model = (w isa AbstractVector) ? @view(w[model_sample_indices]) : w      # w might be nothing

        # Train the model
        model = model_wrapper(X_model, y_model, w_model; 
                                featurenames,
                                rng = rng, 
                                iteration = model_num, 
                                num_models = num_models,
                                class_priors, 
                                kwargs...)
		
        if !isa(model, DecisionList)
            error("The model_wrapper function must return a `DecisionList` instance! got $(typeof(model))")
        end

		models[model_num] = model
	end

    info::NamedTuple = (;
        featurenames,
        supporting_labels=y,
        supporting_predictions=eltype(y)[]
    )

    O = Union{outcometype.(models)...}
    typed_models = Vector{DecisionList{O}}(models)

    return RandomLists(typed_models, info)
end
