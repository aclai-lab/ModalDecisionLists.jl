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
num_kfolds_repeat = 10          # how many times we repeat kfolds

# Execute repeated k-fold cross validation for each target class
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on RIPPER")


results = repeated_cv(
    model_wrapper, metrics_wrapper,
    X, y; 
    rng = rng, 
    num_folds = num_folds, 
    num_repeats = num_kfolds_repeat, 
    loss_function = ModalDecisionLists.LossFunctions.FOILGain(),
    tdl_threshold = 64,
    min_rule_coverage = 2,
    beam_width=10,
    invert_class_orders=false,
    split_ratio=1.0,
    max_k=5,
    verbosity=1
)

print_statistics(results)
