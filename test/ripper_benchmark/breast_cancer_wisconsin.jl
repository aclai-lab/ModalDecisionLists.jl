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
include("helper_functions.jl")

# PART 1 - LOAD THE DATASET
table = dataset("MASS", "biopsy")
y = table[:, :Class] |> CategoricalArray
X = select(table, Not([:ID, :Class]))

X, y = preprocess_inputdata(X,y)
X = DataFrame(X)


# DEFINE RNG FOR REPRODUCIBILITY
rng = Xoshiro(42)
# rng = Random.default_rng()



# DEFINE REPEATED CROSS-VALIDATION PARAMETERS 
num_samples = length(y)
num_folds = 10
num_kfolds_repeat = 2          # how many times we repeat kfolds

println("Num samples: $num_samples")


# PERFORMING CROSS VALIDATION
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on IREP*")
results = stratified_repeated_cv(
    model_wrapper, metrics_wrapper,
    X, y; 
    rng = rng, 
    num_folds = num_folds, 
    num_repeats = num_kfolds_repeat, 
    verbosity=2,

    # RIPPER arguments
    loss_function = ModalDecisionLists.LossFunctions.LaplaceAccuracy(),
    discretizedomain=false,
    min_rule_coverage = 2,
    beam_width = 10,
    tdl_threshold=64,
    split_ratio = 1.0,
    invert_class_orders=false
)

print_statistics(results)