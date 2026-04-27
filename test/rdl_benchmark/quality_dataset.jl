using SoleBase: CLabel
using DataFrames
using SoleModels: apply, DecisionList, solemodel, info, models, weighted_aggregation
using SoleModels
using SoleData
using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray
using StatsBase
using Statistics
using Distributions
using Random
using ModalDecisionLists
using ModalDecisionLists: preprocess_inputdata
using CSV

include("../cv_utilities.jl")
include("test_functions.jl")

# Create a logger to set it in debug mode
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)


# Load the dataset
table = CSV.read("test/datasets/wine_quality.csv", DataFrame)
y = table[:, :quality] |> CategoricalArray
X = select(table, Not([:Id, :quality]));

X, y = preprocess_inputdata(X,y)
X = DataFrame(X)


# rng = Xoshiro(42)
rng = Random.default_rng()

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_kfolds_repeat = 10          # how many times we repeat kfolds

unique_labels = unique(y)

# Execute repeated k-fold cross validation for each model to be tested 
for num_models ∈ [11]

    println("Repeating experiments through k-fold cross validation $num_kfolds_repeat times with k = $num_folds on RDL with $num_models models")

    for lists_perc ∈ [0.0, 1.0]

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
            num_lists = num_lists,
            beam_width = 10,
            suppress_parity_warning = true
        )

        println("\t\t=== 95% confidence intervals ===")
        println("\t\t\tTraining accuracy: $(round(results[:train_accuracy].mean; digits=4)) ± $(round(results[:train_accuracy].margin, digits=4))")
        println("\t\t\tTesting accuracy: $(round(results[:test_accuracy].mean; digits=4)) ± $(round(results[:test_accuracy].margin, digits=4))")

    end


end


X_p = PropositionalLogiset(X)
model_lists = rdl_model_wrapper(X_p, string.(y), rng; 
                                num_models = 5, 
                                num_lists = 5,
                                beam_width = 8)

model_trees = rdl_model_wrapper(X_p, string.(y), rng; 
                                num_models = 5, 
                                num_lists = 0)

println("\n\nLists model: \n$model_lists")
println("\n\nTrees model: \n$model_trees")



