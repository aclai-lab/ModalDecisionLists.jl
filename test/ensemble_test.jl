using SoleBase: CLabel
using DataFrames
using SoleModels: apply, DecisionList, solemodel, info, models
using SoleModels
using SoleData
using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray
using RDatasets
using StatsBase
using Statistics
using Distributions
using Random
using ModalDecisionLists


include("rdl_benchmark/test_functions.jl")      # needed for apply_ensemble

X,y = @load_crabs
X = DataFrame(X) |> PropositionalLogiset
y = string.(y)

rng = Xoshiro(42)

train_ratio = 0.7

train, test = partition(eachindex(y), train_ratio; shuffle=true, rng)
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]


# function called to train the actual base models in the list
model_wrapper(X, y, w; rng, iteration, kwargs...) = ripperk(X, y, w; max_k = 1, rng = rng, kwargs...)

ensemble_model = build_ensemble(X_test, y_test, 11; model_wrapper = model_wrapper)

ensemble_preds = apply_ensemble(ensemble_model, X_test);
ensemble_accuracy = mean(ensemble_preds .== y_test)
println("Ensemble test accuracy: $ensemble_accuracy");

model_alphabet = alphabet(ensemble_model);

println("Model alphabet:\n$model_alphabet");