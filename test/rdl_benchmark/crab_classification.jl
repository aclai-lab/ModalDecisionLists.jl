using SoleBase: CLabel
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
using ModalDecisionLists.Metrics: binary_accuracy
using Logging

include("../cv_utilities.jl")

# Create a logger to set it in debug mode
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)

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
    samples_ratio_per_model = (num_models == 1) ? 1.0 : 1.0
    n_subfeatures_per_model = (num_models == 1) ? nothing : 3
    use_bootstrapping = (num_models != 1)

    rdl_ensemble = build_rdl(
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
        num_lists = num_lists
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
            preds = apply(model, X_model)
        else
            preds = apply(model, X)
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
    model_train_preds = apply_ensemble(model, X_train)
    model_test_preds = apply_ensemble(model, X_test)

    train_accuracy = mean(model_train_preds .== y_train)
    test_accuracy = mean(model_test_preds .== y_test)

    return Dict(
        :train_accuracy => train_accuracy,
        :test_accuracy => test_accuracy
    )
end




# Load the dataset
X, y = @load_crabs
X = DataFrame(X)

# rng = Xoshiro(42)
rng = Random.default_rng()

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_kfolds_repeat = 10          # how many times we repeat kfolds

unique_labels = unique(y)

# Execute repeated k-fold cross validation for each model to be tested 
for num_models ∈ [15]

    println("Repeating experiments through k-fold cross validation $num_kfolds_repeat times with k = $num_folds on RDL with $num_models models")

    for lists_perc ∈ [0.0, 0.5, 1.0]

        num_lists = round(Integer, num_models * lists_perc)

        println("\tTraining model with $num_lists lists and $(num_models - num_lists) decision trees")

        results = repeated_cv(
            rdl_model_wrapper, compute_metrics,
            X, y; 
            rng = rng, 
            num_folds = num_folds, 
            num_repeats = num_kfolds_repeat, 

            # number of models
            num_models = num_models,
            num_lists = num_lists
        )

        println("\t\t=== 95% confidence intervals ===")
        println("\t\t\tTraining accuracy: $(round(results[:train_accuracy].mean; digits=4)) ± $(round(results[:train_accuracy].margin, digits=4))")
        println("\t\t\tTesting accuracy: $(round(results[:test_accuracy].mean; digits=4)) ± $(round(results[:test_accuracy].margin, digits=4))")

    end


end