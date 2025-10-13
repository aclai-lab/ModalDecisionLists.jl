using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, parse_orange_decision_list
using SoleData
using MLJ
using StatsBase
using Random
using ModalDecisionLists


X...,y = MLJ.load_iris()
X_df = DataFrame(X)
X = PropositionalLogiset(X_df)
n_instances = ninstances(X)
y = Vector{CLabel}(y)

sole_decisionlist = IREP_Star(X, y, "setosa")
print(sole_decisionlist isa DecisionList)
sole_outcome_on_training = apply(sole_decisionlist, X)

println(sole_decisionlist)