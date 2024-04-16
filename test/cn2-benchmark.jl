using Test
using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, orange_decision_list
using SoleData
using MLJ
using StatsBase
using Random
using ModalDecisionLists
using ModalDecisionLists: BaseCN2

using BenchmarkTools

# Input
X..., y = MLJ.load_iris()
X_df = DataFrame(X)
X = PropositionalLogiset(X_df)
n_instances = ninstances(X)
y = Vector{CLabel}(y)

# Time
display(@benchmark BaseCN2.build_base_cn2(X_df, y))
display(@benchmark ModalDecisionLists.build_cn2(X, y))


# @btime BaseCN2.build_base_cn2(X_df, y)
# @btime build_cn2(X, y)

# @test_broken outcome_on_training = apply(decision_list, X_pl_view)
