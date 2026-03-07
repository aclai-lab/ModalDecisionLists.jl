using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, parse_orange_decision_list
using SoleData
using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray
using RDatasets
using StatsBase
using Random
using ModalDecisionLists
using ModalDecisionLists.Metrics: binary_accuracy
using Logging

# Create a logger to set it in debug mode
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

# Load the dataset
X,y = @load_iris
X = DataFrame(X)

train_ratio = 0.7
rng = Xoshiro(1)

train, test = partition(eachindex(y), train_ratio; shuffle=true, rng)
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]

println("Training set size: ", size(X_train), " - ", size(y_train))
println("Test set size: ", size(X_test), " - ", size(y_test))
println("Training set type: ", typeof(X_train), " - ", typeof(y_train))
println("Test set type: ", typeof(X_test), " - ", typeof(y_test))

X_train = PropositionalLogiset(X_train)
y_train = String.(y_train)

X_test = PropositionalLogiset(X_test)
y_test = String.(y_test)


target_class = "setosa"
for target_class ∈ ["setosa", "virginica", "versicolor"]
    sole_decisionlist = irepstar(X_train, y_train, target_class, min_rule_coverage = 3)


    # Check performance on training data
    sole_outcome_on_training = apply(sole_decisionlist, X_train)              # Vector{String}
    acc_train = binary_accuracy(y_train, sole_outcome_on_training, target_class)
    println("Model accuracy on the training set for target class $target_class: $acc_train")

    # Check performance on test data 
    sole_outcome_on_test = apply(sole_decisionlist, X_test)
    acc_test = binary_accuracy(y_test, sole_outcome_on_test, target_class)
    println("Model accuracy on the test set for label $target_class: $acc_test")

    println("Decision list obtained for label $target_class: \n{$sole_decisionlist}\n\n")
end