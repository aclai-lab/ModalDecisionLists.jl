using Test

using SoleBase: CLabel
using SoleRules
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, orange_decision_list
using SoleData
using MLJ
using StatsBase
using Random

module BaseCN2
include("../src/algorithms/base-cn2.jl")
end

module SoleCN2
include("../src/algorithms/sole-cn2.jl")
end

# Input
X...,y = MLJ.load_iris()
X_df = DataFrame(X)
X = PropositionalLogiset(X_df)
n_instances = ninstances(X)
y = Vector{CLabel}(y)
############################################################################################


# Test
base_decisionlist = BaseCN2.base_cn2(X_df, y)
sole_decisionlist = SoleCN2.sole_cn2(X, y)

@test base_decisionlist isa DecisionList
@test sole_decisionlist isa DecisionList

base_outcome_on_training = apply(base_decisionlist, X)
sole_outcome_on_training = apply(sole_decisionlist, X)

@test all(base_outcome_on_training .== y)
@test all(sole_outcome_on_training .== y)


orange_decisionlist = """
[1, 0, 0]  IF sepal length<=4.3 THEN iris=Iris-setosa -0.0
[3, 0, 0]  IF sepal length<=4.4 THEN iris=Iris-setosa -0.0
[1, 0, 0]  IF sepal length<=4.5 THEN iris=Iris-setosa -0.0
[4, 0, 0]  IF sepal length<=4.6 THEN iris=Iris-setosa -0.0
[2, 0, 0]  IF sepal length<=4.7 THEN iris=Iris-setosa -0.0
[5, 0, 0]  IF sepal length<=4.8 THEN iris=Iris-setosa -0.0
[0, 0, 12]  IF sepal length>=7.1 THEN iris=Iris-virginica -0.0
[0, 1, 0]  IF sepal length>=7.0 THEN iris=Iris-versicolor -0.0
[0, 1, 0]  IF sepal width<=2.0 THEN iris=Iris-versicolor -0.0
[20, 0, 0]  IF sepal width>=3.5 THEN iris=Iris-setosa -0.0
[1, 0, 0]  IF petal length<=1.2 THEN iris=Iris-setosa -0.0
[3, 0, 0]  IF petal length<=1.4 THEN iris=Iris-setosa -0.0
[6, 0, 0]  IF petal length<=1.5 THEN iris=Iris-setosa -0.0
[2, 0, 0]  IF petal length<=1.6 THEN iris=Iris-setosa -0.0
[2, 0, 0]  IF petal length<=1.7 THEN iris=Iris-setosa -0.0
[0, 1, 0]  IF petal length<=3.0 THEN iris=Iris-versicolor -0.0
[0, 2, 0]  IF petal length<=3.3 THEN iris=Iris-versicolor -0.0
[0, 0, 1]  IF sepal length<=4.9 THEN iris=Iris-virginica -0.0
[0, 1, 0]  IF sepal length<=5.2 THEN iris=Iris-versicolor -0.0
[0, 1, 0]  IF sepal length<=5.4 THEN iris=Iris-versicolor -0.0
[0, 5, 0]  IF sepal length<=5.5 THEN iris=Iris-versicolor -0.0
[0, 1, 0]  IF petal length<=3.5 THEN iris=Iris-versicolor -0.0
[0, 1, 0]  IF petal length<=3.6 THEN iris=Iris-versicolor -0.0
[0, 2, 0]  IF petal length<=3.9 THEN iris=Iris-versicolor -0.0
[0, 3, 0]  IF petal length<=4.0 THEN iris=Iris-versicolor -0.0
[0, 3, 0]  IF petal length<=4.1 THEN iris=Iris-versicolor -0.0
[0, 4, 0]  IF petal length<=4.2 THEN iris=Iris-versicolor -0.0
[0, 2, 0]  IF petal length<=4.3 THEN iris=Iris-versicolor -0.0
[0, 3, 0]  IF petal length<=4.4 THEN iris=Iris-versicolor -0.0
[0, 6, 0]  IF petal length<=4.5 THEN iris=Iris-versicolor -0.0
[0, 0, 1]  IF sepal length<=5.6 THEN iris=Iris-virginica -0.0
[0, 0, 1]  IF sepal length<=5.7 THEN iris=Iris-virginica -0.0
[0, 0, 3]  IF sepal length<=5.8 THEN iris=Iris-virginica -0.0
[0, 0, 1]  IF sepal width<=2.2 THEN iris=Iris-virginica -0.0
[0, 0, 2]  IF sepal width>=3.4 THEN iris=Iris-virginica -0.0
[0, 3, 0]  IF petal length<=4.6 THEN iris=Iris-versicolor -0.0
[0, 4, 0]  IF petal length<=4.7 THEN iris=Iris-versicolor -0.0
[0, 0, 3]  IF sepal width>=3.3 THEN iris=Iris-virginica -0.0
[0, 0, 17]  IF petal length>=5.2 THEN iris=Iris-virginica -0.0
[0, 1, 0]  IF petal width<=1.4 THEN iris=Iris-versicolor -0.0
[0, 0, 3]  IF petal width>=1.9 THEN iris=Iris-virginica -0.0
[0, 2, 0]  IF sepal length>=6.7 THEN iris=Iris-versicolor -0.0
[0, 1, 0]  IF sepal width<=2.5 THEN iris=Iris-versicolor -0.0
[0, 0, 4]  IF sepal length>=6.1 THEN iris=Iris-virginica -0.0
[0, 1, 0]  IF sepal width<=2.7 THEN iris=Iris-versicolor -0.0
[0, 0, 1]  IF sepal length>=6.0 THEN iris=Iris-virginica -0.0
[0, 0, 1]  IF sepal width<=3.0 THEN iris=Iris-virginica -0.0
[0, 1, 0]  IF sepal length<=5.9 THEN iris=Iris-versicolor -0.0
[50, 50, 50]  IF TRUE THEN iris=Iris-virginica -1.584962500721156
"""
imported_decisionlist = SoleModels.orange_decision_list(orange_decisionlist, true)

@test length(listrules(sole_decisionlist)) == length(listrules(base_decisionlist))
@test length(listrules(sole_decisionlist)) == length(listrules(imported_decisionlist))

# TODO da cambiare test
antpairs = zip(SoleModels.antecedent.(rulebase(imported_decisionlist)),
                        SoleModels.antecedent.(rulebase(sole_decisionlist)))

@test [checkconditionsequivalence(i_ant, s_ant) for (i_ant, s_ant) in antpairs] |> all




# @test SoleModels.antecedent.(listrules(sole_decisionlist)) == SoleModels.antecedent.(listrules(base_decisionlist))
# @test SoleModels.consequent.(listrules(sole_decisionlist)) == SoleModels.consequent.(listrules(sole_decisionlist))

# @test (listrules(sole_decisionlist)) == (listrules(base_decisionlist))

############################################################################################

# # Accuracy
# rng = Random.MersenneTwister(1)
# permutation = randperm(rng, n_instances)

# ntrain = round(Int, n_instances/3)*2

# train_slice = permutation[1:ntrain]
# test_slice = permutation[(ntrain+1):end]

# X_train = SoleData.instances(X, train_slice, Val(false))
# X_test = SoleData.instances(X, test_slice, Val(false))

# # ================================
# decisionlist = SoleCN2.sole_cn2(X_train, y[train_slice])
# outcomes = apply(decisionlist, X_test)

# decisionlist2 = BaseCN2.base_cn2(SoleData.gettable(X_train), y[train_slice])
# outcomes2 = apply(decisionlist2, X_test)

# @test all(outcomes .== outcomes2)
# # ================================




# @test_nowarn listrules(decisionlist)
# @test_nowarn listrules(decisionlist2)



# @test MLJ.accuracy(outcomes, y[test_slice]) > 0.8
# @test MLJ.accuracy(outcomes2, y[test_slice]) > 0.8

# ############################################################################################

# using BenchmarkTools
# # Time
# @btime BaseCN2.base_cn2(X_df, y)
# @btime SoleCN2.sole_cn2(X, y)

# @test_broken outcome_on_training = apply(decision_list, X_pl_view)
