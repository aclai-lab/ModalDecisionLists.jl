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

# Create a logger to set it in debug mode
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)

# Load the dataset
X, y = @load_crabs
X = DataFrame(X)

rng = Xoshiro(42)

# folds for cross validation
num_samples = length(y)
num_folds = 10
num_samples_per_fold = num_samples ÷ num_folds      # integer division
num_kfolds_repeat = 10          # how many times we repeat kfolds

unique_labels = ["B", "O"]

# Execute repeated k-fold cross validation
println("Performing repeated k-fold cross validation $num_kfolds_repeat times with k = $num_folds on IREP*")

# total accuracies for each repetition
train_accs = Dict(label => Vector{Float64}(undef, num_kfolds_repeat) for label in unique_labels)
test_accs = Dict(label => Vector{Float64}(undef, num_kfolds_repeat) for label in unique_labels)

# each execution does 1 cross validation run with 'num_folds' folds
for kfold_repeat = 1 : num_kfolds_repeat
    # shuffle everything randomly
    shuffled_indices = randperm(rng, num_samples)
    X_shuffled = X[shuffled_indices, :]
    y_shuffled = y[shuffled_indices, :]

    # history of test and training accuracies for this k-fold cross-validation execution
    cv_train_acc_history = Dict(label => Vector{Float64}(undef, num_folds) for label in unique_labels)
    cv_test_acc_history = Dict(label => Vector{Float64}(undef, num_folds) for label in unique_labels)

    # for each fold for 1 ... k
    for fold_idx = 1 : num_folds
        # Define the "test window"
        fold_data_start = (fold_idx - 1) * num_samples_per_fold + 1
        fold_data_end = min(fold_idx * num_samples_per_fold, num_samples)

        # Extract test set using that window
        X_test = @view X_shuffled[fold_data_start : fold_data_end, :]
        y_test = @view y_shuffled[fold_data_start : fold_data_end]

        # Extract training set (Everything before and after the Window), using (fold_data_start - 1) and (fold_data_end + 1) to avoid overlap
        train_indices = vcat(1 : (fold_data_start - 1), (fold_data_end + 1) : num_samples)
        
        X_train = @view X_shuffled[train_indices, :]
        y_train = @view y_shuffled[train_indices]

        # Convert everything to a Sole PropositionalLogiset
        X_train = PropositionalLogiset(X_train)
        y_train = String.(y_train)

        X_test = PropositionalLogiset(X_test)
        y_test = String.(y_test)

        for target_class ∈ unique_labels
            sole_decisionlist = ripperk(X_train, y_train, target_class, min_rule_coverage = 3; rng = rng, max_k = 1)

            # Check performance on training data
            sole_outcome_on_training = apply(sole_decisionlist, X_train)
            acc_train = binary_accuracy(y_train, sole_outcome_on_training, target_class)
            cv_train_acc_history[target_class][fold_idx] = acc_train

            # Check performance on test data
            sole_outcome_on_test = apply(sole_decisionlist, X_test)
            acc_test = binary_accuracy(y_test, sole_outcome_on_test, target_class)
            cv_test_acc_history[target_class][fold_idx] = acc_test
        end
    end

    for target_class ∈ unique_labels
        # Compute average accuracies
        avg_fold_acc_train = mean(cv_train_acc_history[target_class])
        avg_fold_acc_test = mean(cv_test_acc_history[target_class])

        train_accs[target_class][kfold_repeat] = avg_fold_acc_train
        test_accs[target_class][kfold_repeat] = avg_fold_acc_test
    end
end

for target_class ∈ unique_labels
    avg_acc_train = mean(train_accs[target_class])
    avg_acc_test = mean(test_accs[target_class])

    train_stderr = std(train_accs[target_class]) / sqrt(num_kfolds_repeat)
    test_stderr = std(test_accs[target_class]) / sqrt(num_kfolds_repeat)

    # compute t distribution parameter based on degrees of freedom
    td = TDist(num_kfolds_repeat - 1)
    train_margin = quantile(td, 0.975) * train_stderr
    test_margin = quantile(td, 0.975) * test_stderr

    # compute confidence interval margins
    acc_train_interval_length = train_margin * train_stderr
    acc_test_interval_length = test_margin * test_stderr    

    println("\t=== 95% confidence intervals for target class $target_class (IREP*) ===")
    println("\t\tTraining accuracy: $(round(avg_acc_train; digits=4)) ± $(round(acc_train_interval_length, digits=4))")
    println("\t\tTesting accuracy: $(round(avg_acc_test; digits=4)) ± $(round(acc_test_interval_length, digits=4))")
end