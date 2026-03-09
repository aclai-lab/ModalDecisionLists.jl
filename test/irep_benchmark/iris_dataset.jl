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
using Printf
using Statistics

# Create a logger and set it in debug mode
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)

# Load the dataset
X,y = @load_iris
X = DataFrame(X)

train_ratio = 0.7
rng = Xoshiro(1)        # maybe pass a rand seed to irepstar as well?


# println("Training set size: ", size(X_train), " - ", size(y_train))
# println("Test set size: ", size(X_test), " - ", size(y_test))
# println("Training set type: ", typeof(X_train), " - ", typeof(y_train))
# println("Test set type: ", typeof(X_test), " - ", typeof(y_test))


num_experiments = 100

target_class = "setosa"
for target_class ∈ ["setosa", "virginica", "versicolor"]
    train, test = partition(eachindex(y), train_ratio; shuffle=true, rng)
    X_train, y_train = X[train, :], y[train]
    X_test, y_test = X[test, :], y[test]

    X_train = PropositionalLogiset(X_train)
    y_train = String.(y_train)

    X_test = PropositionalLogiset(X_test)
    y_test = String.(y_test)
    
    # preallocate the empty vectors
    train_accuracies = Vector{Float64}(undef, num_experiments)
    test_accuracies = Vector{Float64}(undef, num_experiments)


    for i = 1 : num_experiments
        sole_decisionlist = irepstar(X_train, y_train, target_class, min_rule_coverage = 3)

        # Check performance on training data
        sole_outcome_on_training = apply(sole_decisionlist, X_train)              # Vector{String}
        acc_train = binary_accuracy(y_train, sole_outcome_on_training, target_class)
        train_accuracies[i] = acc_train 
        # println("Model accuracy on the training set for target class $target_class: $acc_train")

        # Check performance on test data 
        sole_outcome_on_test = apply(sole_decisionlist, X_test)
        acc_test = binary_accuracy(y_test, sole_outcome_on_test, target_class)
        test_accuracies[i] = acc_test
        # println("Model accuracy on the test set for label $target_class: $acc_test")

        # println("Decision list obtained for label $target_class: \n{$sole_decisionlist}\n\n")
    end


    train_acc_mean = mean(train_accuracies)
    train_acc_std = std(train_accuracies)
    train_accuracy_interval_length = 1.96 * train_acc_std / sqrt(num_experiments)

    test_acc_mean = mean(test_accuracies)
    test_acc_std = std(test_accuracies)
    test_accuracy_interval_length = 1.96 * test_acc_std / sqrt(num_experiments)

    println("\n=== Results for 95% confidence intervals after $num_experiments experiments ===")
    @printf("accuracy on training set for target class %s: %.3f ± %.3f\n", target_class, train_acc_mean, train_accuracy_interval_length)
    @printf("accuracy on test set for target class %s: %.3f ± %.3f\n\n", target_class, test_acc_mean, test_accuracy_interval_length)
end