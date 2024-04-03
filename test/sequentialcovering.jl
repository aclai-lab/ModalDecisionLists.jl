using MLJ: load_iris
using DataFrames
using Random
using Test

# using Logging
using SoleBase: CLabel
using ModalDecisionLists
import ModalDecisionLists.SoleCN2: maptointeger


# Iris dataset
X...,y = load_iris()

X = DataFrame(X) |> PropositionalLogiset

y_clabel = Vector{CLabel}(y)
y_string = Vector{String}(y)
y_intger = maptointeger(y)

n_instances = ninstances(X)

w = rand(Float16, n_instances)

############################################################################################
######## empty/mismatch TABLE/TARGET/WEIGHTS ###############################################
############################################################################################

X_empty = DataFrame(Float64[],Float64[]) |> PropositionalLogiset

y_empty = CLabel[]
y_mstch = y_clabel[1:20]

w_empty = Float16[]
w_mstch = w[1:20]


@test_throws ErrorException SoleCN2.sequentialcovering(X_empty,   y_empty)
@test_throws ErrorException SoleCN2.sequentialcovering(X_empty,   y_clabel)
@test_throws ErrorException SoleCN2.sequentialcovering(X,         y_empty)
@test_throws ErrorException SoleCN2.sequentialcovering(X,         y_mstch)
@test_throws ErrorException SoleCN2.sequentialcovering(X,         y_clabel,        w_empty)
@test_throws ErrorException SoleCN2.sequentialcovering(X,         y_clabel,        w_mstch)

############################################################################################
######## PropositionalLogiset, Target ######################################################
############################################################################################

@test_nowarn SoleCN2.sequentialcovering(X, y_clabel)
@test_nowarn SoleCN2.sequentialcovering(X, y_intger)
@test_nowarn SoleCN2.sequentialcovering(X, y_string)

oneinst_X = slicedataset(X, 1; return_view = true)
oneinst_y = y_clabel[1:1]
@test_nowarn SoleCN2.sequentialcovering(oneinst_X, oneinst_y)

############################################################################################
######## PropositionalLogiset, Target, Weights #############################################
############################################################################################

@test_nowarn SoleCN2.sequentialcovering(X, y_clabel, w)
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel, :default)
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel, :rebalance)


@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; beam_width=5)
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=BeamSearch(), beam_width=5)
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=RandSearch(), cardinality=5)

@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=5))
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=RandSearch(; cardinality=5))


bs5 = BeamSearch(; beam_width=5)

@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=bs5, evaluator=soleentropy)
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=5), beam_width=10)
@test_nowarn SoleCN2.sequentialcovering(X, y_clabel; searchmethod=RandSearch(; cardinality=5), cardinality=10)



############################################################################################
######### PropositionalLogiset, Target, beamwidth ##########################################
############################################################################################

@test SoleCN2.sole_cn2(X, y_clabel; beam_width=5) isa DecisionList
@test SoleCN2.sole_cn2(X, y_clabel; beam_width=1) isa DecisionList
@test_throws AssertionError SoleCN2.sole_cn2(X, y_clabel; beam_width=0)

@test SoleCN2.sole_cn2(X, y_clabel; max_rule_length=1000) isa DecisionList
@test SoleCN2.sole_cn2(X, y_clabel; max_rule_length=1) isa DecisionList

@test_throws AssertionError SoleCN2.sole_cn2(X, y_clabel; max_rule_length=0)


############################################################################################
######### PropositionalLogiset, Target, Alphabet ###########################################
############################################################################################

all_alphabet = slicedataset(X,collect(1:n_instances)) |> alphabet
large_alphabet = slicedataset(X,[1,2,3,51,52,53,101,102,103]) |> alphabet
small_alphabet = slicedataset(X,[1,2]) |> alphabet

@test SoleCN2.sole_cn2(X, y_clabel,  alphabet=all_alphabet) isa DecisionList
@test SoleCN2.sole_cn2(X, y_clabel,  alphabet=large_alphabet) isa DecisionList
@test SoleCN2.sole_cn2(X, y_clabel,  alphabet=small_alphabet) isa DecisionList
