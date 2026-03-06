using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, parse_orange_decision_list
using SoleData
# # using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray
using RDatasets
using StatsBase
using Random
using ModalDecisionLists
using Logging


# Create a logger to set it in debug mode
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)


iris = dataset("datasets", "iris")

y = iris[:, :Species] |> CategoricalArray
X = select(iris, Not(:Species))
X = PropositionalLogiset(X)
y = String.(y)


sole_decisionlist = irepstar(X, y, "setosa", min_rule_coverage = 3)
sole_outcome_on_training = apply(sole_decisionlist, X)              # Vector{String}

n = length(y)
println("\tIndex - True label - Pred label")


for i = 1:n 
    pred = sole_outcome_on_training[i]
    correct = y[i]
    println("\t $i \t $correct \t $pred")
end
