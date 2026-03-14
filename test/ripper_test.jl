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
using Test

# Create a logger to set it in debug mode
std_logger = global_logger()
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

target_class = "virginica"

X_train = PropositionalLogiset(X_train)
y_train = String.(y_train)

X_test = PropositionalLogiset(X_test)
y_test = String.(y_test)

result = ripperk(X_train, y_train, target_class; min_rule_coverage = 3)