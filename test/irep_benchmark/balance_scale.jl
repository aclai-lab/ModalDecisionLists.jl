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

# PART 1 - LOAD THE DATASTE
table = CSV.read("test/datasets/balance_scale.csv", DataFrame)
y = table[:, :class] |> CategoricalArray
X = select(table, Not([:class]))

X, y = preprocess_inputdata(X,y)
X = DataFrame(X)
y = string.(y)



# DEFINE RNG FOR REPRODUCIBILITY
rng = Xoshiro(42)
# rng = Random.default_rng()



# DEFINE REPEATED CROSS-VALIDATION PARAMETERS 
num_samples = length(y)
num_folds = 4
num_samples_per_fold = num_samples ÷ num_folds      # integer division
num_kfolds_repeat = 10          # how many times we repeat kfolds



# PERFORMING CROSS VALIDATION
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on IREP*")
results = repeated_cv(
    model_wrapper, metrics_wrapper,
    X, y; 
    rng = rng, 
    num_folds = num_folds, 
    num_repeats = num_kfolds_repeat, 
    loss_function = ModalDecisionLists.LossFunctions.LaplaceAccuracy(),
    min_rule_coverage = 2,
    tdl_threshold=1000000,
    split_ratio = 1.0,
    invert_class_orders = true
)

print_statistics(results)



# SPECIFIC - TRAINING OUTCOME VISUALIZATION

X_prop = PropositionalLogiset(X)

sq_list = sequentialcovering(X_prop, y; min_rule_coverage=3)

y_pred = apply(sq_list, X_prop)
acc_sq = mean(y_pred .== y)
println("Lista con sequential covering:\n$sq_list")

irep_list = irepstar(X_prop, y; rng = rng, tdl_threshold = 128, 
                    min_rule_coverage=2, split_ratio=1.0, invert_class_orders=true, 
                    loss_function = ModalDecisionLists.LossFunctions.LaplaceAccuracy())
y_pred = apply(irep_list, X_prop)
acc_irep = mean(y_pred .== y)
println("\n\nLista con irep*:\n$irep_list")

println("Accuracy lista sequential covering: $acc_sq")
println("Accuracy lista irep: $acc_irep")

