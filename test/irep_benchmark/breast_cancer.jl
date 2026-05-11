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

# PART 1 - LOAD THE DATASET - found at https://archive.ics.uci.edu/dataset/14/breast+cancer and preprocessed
table = CSV.read("test/datasets/breast_cancer.csv", DataFrame)
y = table[:, :Class] |> CategoricalArray
X = select(table, Not([:Class]))

X = DataFrame(X)
y = string.(y)



# DEFINE RNG FOR REPRODUCIBILITY
rng = Xoshiro(42)
# rng = Random.default_rng()



# DEFINE REPEATED CROSS-VALIDATION PARAMETERS 
num_samples = length(y)
num_folds = 10
num_samples_per_fold = num_samples ÷ num_folds      # integer division
num_kfolds_repeat = 2          # how many times we repeat kfolds



# PERFORMING CROSS VALIDATION
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on IREP*")
results = stratified_repeated_cv(
    model_wrapper, metrics_wrapper,
    X, y; 
    rng = rng, 
    num_folds = num_folds, 
    num_repeats = num_kfolds_repeat, 
    verbosity=1,

    # IREP* arguments
    loss_function = ModalDecisionLists.LossFunctions.FOILGain(),
    discretizedomain=false,
    min_rule_coverage = 2,
    beam_width = 30,
    tdl_threshold=64,
    split_ratio = 1.0,
    invert_class_orders=true
)

print_statistics(results)