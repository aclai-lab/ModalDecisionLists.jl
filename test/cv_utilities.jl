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

"""
    function repeated_cv(
        model_wrapper::Function, 
        metrics_wrapper::Function,
        X, 
        y; 
        num_folds = 10, 
        num_repeats = 10, 
        rng = Random.default_rng(), 
        kwargs...
    )

Run repeated k-fold cross validation for a model and compute mean metrics with confidence margins.

# Arguments
- `model_wrapper::Function`: A function `(X_train, y_train, rng; kwargs...) -> model` that trains and returns a fitted model.
- `metrics_wrapper::Function`: A function `(model, X_train, y_train, X_test, y_test; kwargs...) -> Dict{Symbol, Float64}` that evaluates the model and returns named metrics.
- `X`: Feature table (typically a DataFrame) with rows as samples.
- `y`: Target vector (categorical labels) aligned with rows of `X`.

# Keyword Arguments
- `num_folds::Int=10`: Number of folds per repetition.
- `num_repeats::Int=10`: Number of repeated cross-validation iterations.
- `rng`: Random number generator used for shuffling.
- `kwargs...`: Additional keyword args forwarded to `model_wrapper` and `metrics_wrapper`.

# Returns
A `Dict{Symbol, NamedTuple}` where each key is a metric name and each value is a named tuple with:
- `mean`: the mean metric across repeated fold averages.
- `margin`: the 95% t-distribution margin of error.

# Behavior
For each repetition, data is shuffled, split into `num_folds`, and the model is trained and evaluated on each fold. Per-fold metric values are averaged within each repetition, and final statistics are computed across repetitions.
"""
function repeated_cv(
    model_wrapper::Function, 
    metrics_wrapper::Function,
    X, 
    y; 
    num_folds = 10, 
    num_repeats = 10, 
    rng = Random.default_rng(), 
    use_views::Bool = true,
    verbosity::Integer = 0,
    kwargs...
)
    num_samples = length(y)
    num_samples_per_fold = num_samples ÷ num_folds
    
    results_history = Dict{Symbol, Vector{Float64}}()

    for r in 1:num_repeats
        shuffled_indices = randperm(rng, num_samples)
        X_shf = use_views ? @view(X[shuffled_indices, :]) : X[shuffled_indices, :]
        y_shf = use_views ? @view(y[shuffled_indices]) : y[shuffled_indices]

        fold_metrics = Dict{Symbol, Vector{Float64}}()

        for f in 1:num_folds
            start_i = (f - 1) * num_samples_per_fold + 1
            end_i = (f == num_folds) ? num_samples : f * num_samples_per_fold
            
            # FIX: with 1 fold, train on the full dataset (in-sample evaluation)
            test_idx  = start_i:end_i
            train_idx = num_folds == 1 ? (1:num_samples) : vcat(1:(start_i-1), (end_i+1):num_samples)

            X_train_raw = use_views ? @view(X_shf[train_idx, :]) : X_shf[train_idx, :]
            y_train = string.(use_views ? @view(y_shf[train_idx]) : y_shf[train_idx])

            X_test_raw = use_views ? @view(X_shf[test_idx, :]) : X_shf[test_idx, :]
            y_test = string.(use_views ? @view(y_shf[test_idx]) : y_shf[test_idx])

            X_train = PropositionalLogiset(X_train_raw)
            X_test  = PropositionalLogiset(X_test_raw)

            model = model_wrapper(X_train, y_train, rng; kwargs...)
            metrics = metrics_wrapper(model, X_train, y_train, X_test, y_test; kwargs...)

            for (m_name, m_val) ∈ metrics
                # push!(get!(fold_metrics, m_name, Float64[]), m_val)
                push!(get!(results_history, m_name, Float64[]), m_val)
            end

            if verbosity >= 2
                println("Finished fold #$f")
            end
        end

        
        # for (m_name, m_vals) ∈ fold_metrics
        #     push!(get!(results_history, m_name, Float64[]), mean(m_vals))
        # end

        if verbosity >= 1
            println("Finished repetition #$r")
        end
    end

    final_results = _aggregate_cv_results(results_history, num_repeats * num_folds)
    return final_results
end

function _aggregate_cv_results(results_history, num_repeats)
    use_ci = num_repeats > 1
    t_val  = use_ci ? quantile(TDist(num_repeats - 1), 0.975) : 0.0
    return Dict(
        m_name => (
            mean   = mean(values),
            margin = t_val * (use_ci ? std(values) / sqrt(num_repeats) : 0.0),
            std = std(values)
        )
        for (m_name, values) ∈ results_history
    )
end


function stratified_repeated_cv(
    model_wrapper::Function,
    metrics_wrapper::Function,
    X,
    y;
    num_folds   = 10,
    num_repeats = 10,
    rng         = Random.default_rng(),
    use_views::Bool = true,
    positive_class = nothing,
    verbosity::Integer = 0,
    kwargs...
)
    # --- Validate binary target -----------------------------------------------
    classes = unique(y)
    length(classes) == 2 || throw(ArgumentError(
        "stratified_repeated_cv requires exactly 2 classes, got: $classes"
    ))

    # Validate and resolve which class is positive
    if positive_class === nothing
        class_pos, class_neg = classes[1], classes[2]
    else
        class_pos = positive_class
        class_neg = only(filter(!=(positive_class), classes))
    end

    # Pre-split indices by class (done once; shuffling happens per repetition)
    idx_pos = findall(==(class_pos), y)
    idx_neg = findall(==(class_neg), y)

    n_pos = length(idx_pos)
    n_neg = length(idx_neg)

    # Each fold gets floor(n_pos/num_folds) positives and floor(n_neg/num_folds) negatives.
    # The last fold absorbs any remainder (same strategy as the base function).
    fold_size_pos = n_pos ÷ num_folds
    fold_size_neg = n_neg ÷ num_folds

    (fold_size_pos == 0 || fold_size_neg == 0) && throw(ArgumentError(
        "Too many folds ($num_folds) for class sizes ($n_pos pos, $n_neg neg). " *
        "Reduce num_folds so every fold has at least one sample per class."
    ))

    results_history = Dict{Symbol, Vector{Float64}}()

    for r in 1:num_repeats
        # Shuffle each class independently to preserve stratification
        shf_pos = idx_pos[randperm(rng, n_pos)]
        shf_neg = idx_neg[randperm(rng, n_neg)]

        fold_metrics = Dict{Symbol, Vector{Float64}}()

        for f in 1:num_folds
            # build per-class test slices 
            pos_start = (f - 1) * fold_size_pos + 1
            pos_end   = f == num_folds ? n_pos : f * fold_size_pos

            neg_start = (f - 1) * fold_size_neg + 1
            neg_end   = f == num_folds ? n_neg : f * fold_size_neg

            test_pos  = pos_start:pos_end
            test_neg  = neg_start:neg_end

            # assemble global test / train index vectors
            test_idx  = vcat(shf_pos[test_pos],  shf_neg[test_neg])

            if num_folds == 1
                # In-sample evaluation: train on the full dataset
                train_idx = vcat(shf_pos, shf_neg)
            else
                train_pos = vcat(shf_pos[1:(pos_start - 1)], shf_pos[(pos_end + 1):n_pos])
                train_neg = vcat(shf_neg[1:(neg_start - 1)], shf_neg[(neg_end + 1):n_neg])
                train_idx = vcat(train_pos, train_neg)
            end

            # data slices
            X_train_raw = use_views ? @view(X[train_idx, :]) : X[train_idx, :]
            y_train     = string.(use_views ? @view(y[train_idx]) : y[train_idx])

            X_test_raw  = use_views ? @view(X[test_idx, :]) : X[test_idx, :]
            y_test      = string.(use_views ? @view(y[test_idx]) : y[test_idx])

            # convert to PropositionalLogiset, then train and evaluate
            X_train = PropositionalLogiset(X_train_raw)
            X_test  = PropositionalLogiset(X_test_raw)

            model   = model_wrapper(X_train, y_train, rng; kwargs...)
            metrics = metrics_wrapper(model, X_train, y_train, X_test, y_test; kwargs...)

            for (m_name, m_val) ∈ metrics
                # push!(get!(fold_metrics, m_name, Float64[]), m_val)
                push!(get!(results_history, m_name, Float64[]), m_val)
            end

            if verbosity >= 2
                println("\tFinished fold $f")
            end
        end

        # for (m_name, m_vals) ∈ fold_metrics
        #     push!(get!(results_history, m_name, Float64[]), mean(m_vals))
        # end

        if verbosity >= 1
            println("Finished repetition #$r")
        end
    end

    return _aggregate_cv_results(results_history, num_repeats)
end