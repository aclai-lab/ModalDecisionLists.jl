using MLJ: load_iris
using DataFrames
using Random
using Test
using CategoricalArrays
# using Logging
using SoleBase
using SoleBase: CLabel
using ModalDecisionLists
import ModalDecisionLists: maptointeger

import ModalDecisionLists.Measures: entropy, laplace_accuracy

# Iris dataset
X...,y = load_iris()

X = DataFrame(X) |> PropositionalLogiset

y_clabel = Vector{CLabel}(y)
y_string = Vector{String}(y)
y_catgcl = CategoricalArray(y)
y_intger, _ = maptointeger(y)

n_instances = ninstances(X)

w = rand(Float16, n_instances)

############################################################################################
######## empty/mismatch TABLE/TARGET/WEIGHTS ###############################################
############################################################################################

y_empty = CLabel[]
y_mstch = y_clabel[1:20]

w_empty = Float16[]
w_mstch = w[1:20]

@test_throws ErrorException sequentialcovering(X,         y_empty)
@test_throws ErrorException sequentialcovering(X,         y_mstch)
@test_throws ErrorException sequentialcovering(X,         y_clabel,   w_empty)
@test_throws ErrorException sequentialcovering(X,         y_clabel,   w_mstch)

############################################################################################
######## PropositionalLogiset, Target ######################################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel)
@test_nowarn sequentialcovering(X, y_intger)
@test_nowarn sequentialcovering(X, y_string)
@test_nowarn sequentialcovering(X, y_catgcl)


oneinst_X, oneinst_y = slicedataset(X, 1; return_view = true), y_clabel[1:1]
@test_nowarn sequentialcovering(oneinst_X, oneinst_y)

############################################################################################
######## PropositionalLogiset, Target, Weights #############################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel, w)
@test_nowarn sequentialcovering(X, y_clabel, :default)
@test_nowarn sequentialcovering(X, y_clabel, :rebalance)
@test_throws AssertionError sequentialcovering(X, y_clabel, :nomeaning)


@test_nowarn sequentialcovering(X, y_clabel; beam_width=1)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=3)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=5)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=25)

@test_nowarn sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=5))

@test_throws AssertionError sequentialcovering(X, y_clabel; beam_width=0)
@test_throws AssertionError sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=0))
# Mi assicuro che il parametro venga sovrascrito
@test_throws AssertionError sequentialcovering(X, y_clabel; beam_width=0, searchmethod=BeamSearch(; beam_width=5))


@test_throws AssertionError sequentialcovering(X, y_clabel; max_rulebase_length=0)
@test_logs (:warn,"Parity encountered in bestguess! counts (146 elements):" *
            " Dict(2 => 50, 3 => 50, 1 => 46), argmax: 2, max: 50 (sum = 146)"
    ) sequentialcovering(X, y_clabel; max_rulebase_length=1)
@test_nowarn sequentialcovering(X, y_clabel; max_rulebase_length=10)
@test_nowarn sequentialcovering(X, y_clabel; max_rulebase_length=1000)


bs5_ent = BeamSearch(; beam_width=5, quality_evaluator=entropy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs5_ent)

bs_entropy = BeamSearch(; quality_evaluator=entropy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs_entropy)

bs_laplace = BeamSearch(; quality_evaluator=laplace_accuracy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs_laplace)

@test_nowarn sequentialcovering(X, y_clabel; beam_width=1, quality_evaluator=laplace_accuracy)

############################################################################################
######### PropositionalLogiset, Target, Alphabet ###########################################
############################################################################################

# TODO @Edo test for RandSearch


# all_alphabet = slicedataset(X,collect(1:n_instances)) |> alphabet
# large_alphabet = slicedataset(X,[1,2,3,51,52,53,101,102,103]) |> alphabet
# small_alphabet = slicedataset(X,[1,2]) |> alphabet

# @test_nowarn sequentialcovering(X, y_clabel,  alphabet=all_alphabet)
# @test_nowarn sequentialcovering(X, y_clabel,  alphabet=large_alphabet)
# @test_nowarn sequentialcovering(X, y_clabel,  alphabet=small_alphabet)
