using SoleBase: CLabel
using DataFrames
using SoleModels: apply, DecisionList
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
X, y = @load_crabs
X = DataFrame(X)

train_ratio = 0.7
rng = Xoshiro(2)

num_experiments = 30

# Initialize accumulators for accuracies
acc_train_history = Dict("B" => [], "O" => [])
acc_test_history = Dict("B" => [], "O" => [])


for j = 0 : 10
    num_models = 2*j + 1
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

        # Test build_rdl with different target classes
        for target_class ∈ ["B", "O"]
            # println("\n" * "="^80)
            # println("Building RDL ensemble for target class: $target_class")
            # println("="^80)

            # if num_models is 1, just do a standard training with all samples and all features
            samples_ratio_per_model = (num_models == 1) ? 1.0 : 1.0
            n_subfeatures_per_model = (num_models == 1) ? nothing : 3
            use_bootstrapping = (num_models != 1)

            rdl_ensemble = build_rdl(
                X_train, 
                y_train, 
                target_class,
                num_models;
                
                # rdl arguments
                use_bootstrapping = use_bootstrapping,
                samples_ratio_per_model=samples_ratio_per_model,
                n_subfeatures_per_model=n_subfeatures_per_model,
                
                # kwargs passed directly to irep*
                min_rule_coverage=3
            )

            # Check number of models in the ensemble
            # println("Number of decision lists in ensemble: $(nlists(rdl_ensemble))")

            # Check performance on training data
            rdl_outcome_on_training = apply(rdl_ensemble, X_train)
            acc_train = binary_accuracy(y_train, rdl_outcome_on_training, target_class)
            push!(acc_train_history[target_class], acc_train)
            # println("RDL ensemble accuracy on the training set for target class $target_class: $acc_train")

            # Check performance on test data
            rdl_outcome_on_test = apply(rdl_ensemble, X_test)
            acc_test = binary_accuracy(y_test, rdl_outcome_on_test, target_class)
            push!(acc_test_history[target_class], acc_test)
            # println("RDL ensemble accuracy on the test set for target class $target_class: $acc_test")
        end
    end

    # Compute and print average accuracies
    println("\n=== Results after $num_experiments experiments with $num_models models ===")
    for target_class ∈ ["B", "O"]
        avg_acc_train = mean(acc_train_history[target_class])
        avg_acc_test = mean(acc_test_history[target_class])
        println("Target class $target_class:")
        println("  Average training accuracy: $(round(avg_acc_train; digits=4))")
        println("  Average testing accuracy: $(round(avg_acc_test; digits=4))")
    end
end