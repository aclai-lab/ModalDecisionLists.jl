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

iris = dataset("datasets", "iris")

y = iris[:, :Species] |> CategoricalArray
X = select(iris, Not(:Species))
X = PropositionalLogiset(X)
y = String.(y)

sole_decisionlist = irepstar(X, y, "setosa", min_rule_coverage = 3)
# @show sole_decisionlist
# sole_outcome_on_training = apply(sole_decisionlist, X)
#
# n = length(y)
# println("Index - True label - Pred label")
#
# for i = 1:n 
#     pred = sole_outcome_on_training[i]
#     correct = y[i]
#     println("\t $i \t $correct \t $pred")
# end
