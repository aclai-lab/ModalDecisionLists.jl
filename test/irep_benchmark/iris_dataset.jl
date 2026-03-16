using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, parse_orange_decision_list
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

# Load the dataset
X, y = @load_iris
X = DataFrame(X)

rng = Xoshiro(42)
# rng = Random.default_rng()

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_samples_per_fold = num_samples ÷ num_folds      # integer division
num_kfolds_repeat = 10          # how many times we repeat kfolds

function model_wrapper(X, y, rng; kwargs...)
    args_dict = Dict(kwargs)
    target_class = pop!(args_dict, :target_class)
    loss_function = pop!(args_dict, :loss_function)
    min_rule_coverage = pop!(args_dict, :min_rule_coverage)

    irepstar(X, y, target_class; rng = rng, loss_function = loss_function, min_rule_coverage = min_rule_coverage, args_dict...)
end

function metrics_wrapper(model, X_train, y_train, X_test, y_test; kwargs...)
    args_dict = Dict(kwargs)
    target_class = pop!(args_dict, :target_class)

    model_train_preds = apply(model, X_train)
    model_test_preds = apply(model, X_test)

    return Dict(
        :train_accuracy => binary_accuracy(y_train, model_train_preds, target_class),
        :test_accuracy => binary_accuracy(y_test, model_test_preds, target_class)
    )
end

unique_labels = ["setosa", "virginica", "versicolor"]

# Execute repeated k-fold cross validation for each target class
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on RIPPER")
for target_class ∈ unique_labels


    results = repeated_cv(
        model_wrapper, metrics_wrapper,
        X, y; 
        rng = rng, 
        num_folds = num_folds, 
        num_repeats = num_kfolds_repeat, 
        target_class = target_class,
        loss_function = ModalDecisionLists.LossFunctions.LaplaceAccuracy(),
        min_rule_coverage = 3
    )


    println("\t=== 95% confidence intervals for target class $target_class ===")
    println("\t\tTraining accuracy: $(round(results[:train_accuracy].mean; digits=4)) ± $(round(results[:train_accuracy].margin, digits=4))")
    println("\t\tTesting accuracy: $(round(results[:test_accuracy].mean; digits=4)) ± $(round(results[:test_accuracy].margin, digits=4))")
end