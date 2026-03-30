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

include("../cv_utilities.jl")
include("test_functions.jl")

# Create a logger to set it in debug mode
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)


# Load the dataset
X, y = @load_iris
X = DataFrame(X)

# rng = Xoshiro(42)
rng = Random.default_rng()

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_kfolds_repeat = 10          # how many times we repeat kfolds

unique_labels = unique(y)

# Execute repeated k-fold cross validation for each model to be tested 
for num_models ∈ [5, 10, 15]

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