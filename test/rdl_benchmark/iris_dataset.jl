using SoleBase: CLabel
using DataFrames
using SoleModels: apply, DecisionList
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

function model_wrapper(X, y, rng; kwargs...)
    args_dict = Dict(kwargs)
    target_class = pop!(args_dict, :target_class)
    num_models = pop!(args_dict, :num_models)

    samples_ratio_per_model = (num_models == 1) ? 1.0 : 1.0
    n_subfeatures_per_model = (num_models == 1) ? nothing : 3
    use_bootstrapping = (num_models != 1)

    rdl_ensemble = build_rdl(
        X, 
        y, 
        target_class,
        num_models;
        
        # rdl arguments
        use_bootstrapping = use_bootstrapping,
        samples_ratio_per_model = samples_ratio_per_model,
        n_subfeatures_per_model = n_subfeatures_per_model,
        rng = rng,
        model_wrapper = ripperk,

        # kwargs passed directly to model_wrapper
        min_rule_coverage=3
    )

    return rdl_ensemble
end

function metrics_wrapper(model, X_train, y_train, X_test, y_test; kwargs...)
    model_train_preds = apply(model, X_train)
    model_test_preds = apply(model, X_test)

    train_accuracy = mean(model_train_preds .== y_train)
    test_accuracy = mean(model_test_preds .== y_test)

    return Dict(
        :train_accuracy => train_accuracy,
        :test_accuracy => test_accuracy
    )
end


# Load the dataset
X, y = @load_iris
X = DataFrame(X)

rng = Xoshiro(42)

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_kfolds_repeat = 10          # how many times we repeat kfolds

unique_labels = unique(y)

# Execute repeated k-fold cross validation for each model to be tested 
for num_models ∈ [1, 5, 11, 21]

    println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on RDL with $num_models models")

    for target_class ∈ unique_labels

        results = repeated_cv(
            model_wrapper, metrics_wrapper,
            X, y; 
            rng = rng, 
            num_folds = num_folds, 
            num_repeats = num_kfolds_repeat, 
            
            # number of models
            num_models = num_models
        )


        println("\t=== 95% confidence intervals for target class $target_class  ===")
        println("\t\tTraining accuracy: $(round(results[:train_accuracy].mean; digits=4)) ± $(round(results[:train_accuracy].margin, digits=4))")
        println("\t\tTesting accuracy: $(round(results[:test_accuracy].mean; digits=4)) ± $(round(results[:test_accuracy].margin, digits=4))")
    end

end