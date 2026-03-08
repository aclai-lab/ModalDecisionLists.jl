using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, parse_orange_decision_list
using SoleData
using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray
using RDatasets
using StatsBase
using Statistics
using Random
using ModalDecisionLists
using ModalDecisionLists.Metrics: binary_accuracy
using Logging

# Create a logger to set it in debug mode
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)

# Load the dataset
X, y = @load_iris
X = DataFrame(X)

train_ratio = 0.7
rng = Xoshiro(1)

num_experiments = 1

# Initialize accumulators for accuracies
acc_train_history = []
acc_test_history = []

for i = 1 : num_experiments
    train, test = partition(eachindex(y), train_ratio; shuffle=true, rng)
    X_train, y_train = X[train, :], y[train]
    X_test, y_test = X[test, :], y[test]

    # println("Training set size: ", size(X_train), " - ", size(y_train))
    # println("Test set size: ", size(X_test), " - ", size(y_test))
    # println("Training set type: ", typeof(X_train), " - ", typeof(y_train))
    # println("Test set type: ", typeof(X_test), " - ", typeof(y_test))

    X_train = PropositionalLogiset(X_train)
    y_train = String.(y_train)

    X_test = PropositionalLogiset(X_test)
    y_test = String.(y_test)

    # Test multiclass irepstar
    sole_decisionlist = irepstar(X_train, y_train, min_rule_coverage = 3)

    # Check performance on training data
    sole_outcome_on_training = apply(sole_decisionlist, X_train)              # Vector{String}
    acc_train = mean(sole_outcome_on_training .== y_train)
    push!(acc_train_history, acc_train)
    # println("Model accuracy on the training set: $acc_train")

    # Check performance on test data 
    sole_outcome_on_test = apply(sole_decisionlist, X_test)
    acc_test = mean(sole_outcome_on_test .== y_test)
    push!(acc_test_history, acc_test)
    # println("Model accuracy on the test set: $acc_test")

    # println("Decision list obtained: \n{$sole_decisionlist}\n\n")
end

# Compute and print average accuracies
println("\n=== Results after $num_experiments experiments (Multiclass IREP*) ===")
avg_acc_train = mean(acc_train_history)
avg_acc_test = mean(acc_test_history)
println("Average training accuracy: $(round(avg_acc_train; digits=4))")
println("Average testing accuracy: $(round(avg_acc_test; digits=4))")
