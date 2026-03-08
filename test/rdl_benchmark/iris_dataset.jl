using SoleBase: CLabel
using DataFrames
using SoleModels: apply, DecisionList
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
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)

# Load the dataset
X, y = @load_iris
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

# Test build_rdl with different target classes
for target_class ∈ ["setosa", "virginica", "versicolor"]
    println("\n" * "="^80)
    println("Building RDL ensemble for target class: $target_class")
    println("="^80)
    num_models = 10

    rdl_ensemble = build_rdl(
        X_train, 
        y_train, 
        target_class,
        5;
        samples_ratio_per_model=0.8,
        min_rule_coverage=3
    )

    # Check number of models in the ensemble
    println("Number of decision lists in ensemble: $(nlists(rdl_ensemble))")

    # Check performance on training data
    rdl_outcome_on_training = apply(rdl_ensemble, X_train)
    acc_train = binary_accuracy(y_train, rdl_outcome_on_training, target_class)
    println("RDL ensemble accuracy on the training set for target class $target_class: $acc_train")

    # Check performance on test data
    rdl_outcome_on_test = apply(rdl_ensemble, X_test)
    acc_test = binary_accuracy(y_test, rdl_outcome_on_test, target_class)
    println("RDL ensemble accuracy on the test set for target class $target_class: $acc_test")
end