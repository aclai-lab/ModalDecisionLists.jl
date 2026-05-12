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
include("helper_functions.jl")


# Load the dataset
X, y = @load_crabs
X = DataFrame(X)

rng = Xoshiro(42)
# rng = Random.default_rng()

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_kfolds_repeat = 10          # how many times we repeat kfolds

unique_labels = unique(y)


# Execute repeated k-fold cross validation for each target class
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on IREP*")

results = stratified_repeated_cv(
    model_wrapper, metrics_wrapper,
    X, y; 
    rng = rng, 
    num_folds = num_folds, 
    num_repeats = num_kfolds_repeat, 
    
    # IREP* parameters
    loss_function = ModalDecisionLists.LossFunctions.LaplaceAccuracy(),
    tdl_threshold = 32,
    split_ratio=0.7,
    beam_width=10,
    min_rule_coverage = 3,
    verbosity=1
)


print_statistics(results)