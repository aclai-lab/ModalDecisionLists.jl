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
    kwargs...
)
    num_samples = length(y)
    num_samples_per_fold = num_samples ÷ num_folds
    
    # Store results for each repetition
    results_history = Dict{Symbol, Vector{Float64}}()

    for r in 1:num_repeats
        # Shuffle indices for this repetition
        shuffled_indices = randperm(rng, num_samples)
        X_shf = use_views ? @view(X[shuffled_indices, :]) : X[shuffled_indices, :]
        y_shf = use_views ? @view(y[shuffled_indices]) : y[shuffled_indices]

        # Temporary storage for fold results to average them for this repetition
        fold_metrics = Dict{Symbol, Vector{Float64}}()

        for f in 1:num_folds
            # Define fold boundaries
            start_i = (f - 1) * num_samples_per_fold + 1
            end_i = min(f * num_samples_per_fold, num_samples)
            
            test_idx = start_i:end_i
            train_idx = vcat(1:(start_i-1), (end_i+1):num_samples)

            # Data Preparation (Views for efficiency)
            X_train_raw = use_views  ?  @view(X_shf[train_idx, :])  :  X_shf[train_idx, :]
            y_train = string.(  use_views ? @view(y_shf[train_idx]) : y_shf[train_idx]  )

            X_test_raw  = use_views  ?  @view(X_shf[test_idx, :])  :  X_shf[test_idx, :]
            y_test  = string.(use_views  ?  @view(y_shf[test_idx])  :  y_shf[test_idx])

            # Convert to Sole format
            X_train = PropositionalLogiset(X_train_raw)
            X_test  = PropositionalLogiset(X_test_raw)

            # Train the model using the wrapper
            model = model_wrapper(X_train, y_train, rng; kwargs...)

            # Evaluate
            metrics = metrics_wrapper(model, X_train, y_train, X_test, y_test; kwargs...)

            # Initialize fold storage if first fold
            for (m_name, m_val) ∈ metrics
                push!(get!(fold_metrics, m_name, Float64[]), m_val)
            end
        end

        # Average the folds for this repetition and push to global history
        for (m_name, m_vals) ∈ fold_metrics
            push!(get!(results_history, m_name, Float64[]), mean(m_vals))
        end
    end

    # Statistical Calculations
    td = TDist(num_repeats - 1)
    t_val = quantile(td, 0.975)

    final_results = Dict{Symbol, NamedTuple}()

    for (m_name, values) ∈ results_history
        avg = mean(values)
        m_stderr = std(values) / sqrt(num_repeats)
        final_results[m_name] = (
            mean = avg,
            margin = t_val * m_stderr,
        )
    end

    return final_results
end