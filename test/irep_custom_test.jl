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

sole_decisionlist = IREP_Star(X, y, "setosa")
print(sole_decisionlist isa DecisionList)
sole_outcome_on_training = apply(sole_decisionlist, X)

println(sole_decisionlist)
