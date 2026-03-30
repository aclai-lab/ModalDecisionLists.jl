using DataFrames
using SoleModels: apply, DecisionList, solemodel, info, models, weighted_aggregation
using SoleModels
using SoleData
using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray
using RDatasets
using StatsBase
using Statistics
using Distributions
using Random
using ModalDecisionLists

DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree


function rdl_tree_wrapper(X, y, w; rng, kwargs...)
    # Prepare data for MLJ
    X_fixed = DataFrame(X)  
    y_fixed = categorical(y) 

    # TODO: actually implement this into machine()
    actual_w = isnothing(w) ? nothing : collect(Float64, w)

    tree_model = DecisionTreeClassifier(rng = rng)
    
    mach = machine(tree_model, X_fixed, y_fixed)

    fit!(mach, verbosity = 0)
    
    trained_features = report(mach).features        # set of features from X

    sole_tree = solemodel(fitted_params(mach).tree; featurenames = trained_features)

    return sole_tree    
end


function split_model_wrapper(X, y, w; rng, iteration, num_lists, kwargs...)
    if iteration <= num_lists
        return ripperk(X, y, w; max_k = 5, rng = rng, kwargs...)
    else
        return rdl_tree_wrapper(X, y, w; rng = rng, kwargs...)  
    end 

end

# Function passed to repeated_cv to train an ensemble model with a number 'num_lists' of DecisionLists, and 'num_models - num_lists' of decision tree
function rdl_model_wrapper(X, y, rng; num_models, num_lists, kwargs...)
    num_features = nfeatures(X)

    samples_ratio_per_model = (num_models == 1) ? 1.0 : 1.0
    n_subfeatures_per_model = (num_models == 1) ? nothing : round(Integer, sqrt(num_features))
    use_bootstrapping = (num_models != 1)

    rdl_ensemble = build_ensemble(
        X, 
        y, 
        num_models;
        
        # rdl arguments
        use_bootstrapping = use_bootstrapping,
        samples_ratio_per_model = samples_ratio_per_model,
        n_subfeatures_per_model = n_subfeatures_per_model,
        rng = rng,
        model_wrapper = split_model_wrapper,

        # kwargs passed directly to split_model_wrapper
        min_rule_coverage=3,
        num_lists = num_lists,

        kwargs...
    )

    return rdl_ensemble
end


function __apply_post(m, preds)
    if haskey(info(m), :apply_postprocess)
        apply_postprocess_f = info(m, :apply_postprocess)
        preds = apply_postprocess_f.(preds)
    end
    preds
end



function apply_ensemble(
    model::SoleModels.DecisionEnsemble,
    X::PropositionalLogiset;
    suppress_parity_warning = false,
    kwargs...
)   
    submodels = models(model)

    total_preds = []

    for model ∈ submodels
        if hasproperty(model, :info) && haskey(info(model), :featurenames)
            feature_names = info(model)[:featurenames]
            X_model = X[:, feature_names]
            preds = apply(model, X_model; suppress_parity_warning = suppress_parity_warning, kwargs...)
        else
            preds = apply(model, X; suppress_parity_warning = suppress_parity_warning, kwargs...)
        end

        push!(total_preds, preds)
    end

    preds = hcat(total_preds...)
    preds = __apply_post(model, preds)
    preds = [
        weighted_aggregation(model)(preds[i,:]; suppress_parity_warning)
        for i in 1:size(preds,1)
    ]
    return preds
end

function compute_metrics(model, X_train, y_train, X_test, y_test; kwargs...)
    model_train_preds = apply_ensemble(model, X_train; suppress_parity_warning = true)
    model_test_preds = apply_ensemble(model, X_test, suppress_parity_warning = true)

    train_accuracy = mean(model_train_preds .== y_train)
    test_accuracy = mean(model_test_preds .== y_test)

    return Dict(
        :train_accuracy => train_accuracy,
        :test_accuracy => test_accuracy
    )
end
