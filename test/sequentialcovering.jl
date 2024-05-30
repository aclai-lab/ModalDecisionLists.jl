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

import ModalDecisionLists.LossFunctions: entropy, laplace_accuracy

# Iris dataset
X...,y = load_iris()

X = DataFrame(X) |> PropositionalLogiset

y_clabel = Vector{CLabel}(y)
y_string = Vector{String}(y)
y_catgcl = CategoricalArray(y)
y_intger, _ = maptointeger(y)
n_instances = ninstances(X)
w = rand(Float16, n_instances)

# function sequentialcovering(
#     X::AbstractLogiset,
#     y::AbstractVector{<:CLabel},
#     w::Union{Nothing,AbstractVector{U},Symbol} = default_weights(length(y));
#     searchmethod::SearchMethod=BeamSearch(),
#     max_rulebase_length::Union{Nothing,Integer}=nothing,
#     suppress_parity_warning::Bool=false,
#     kwargs...
# )::DecisionList where {U<:Real}


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
############################## Target ######################################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel)
@test_nowarn sequentialcovering(X, y_intger)
@test_nowarn sequentialcovering(X, y_string)
@test_nowarn sequentialcovering(X, y_catgcl)

oneinst_X, oneinst_y = slicedataset(X, 1; return_view = true), y_clabel[1:1]
@test_nowarn sequentialcovering(oneinst_X, oneinst_y)

############################################################################################
############################## Weights #####################################################
############################################################################################
@test_nowarn sequentialcovering(X, y_clabel, w)
@test_nowarn sequentialcovering(X, y_clabel, :default)
@test_nowarn sequentialcovering(X, y_clabel, :rebalance)
@test_throws AssertionError sequentialcovering(X, y_clabel, :nomeaning)


############################################################################################
############################## suppress_parity_warning #####################################
############################################################################################

@test_throws AssertionError sequentialcovering(X, y_clabel; max_rulebase_length=0)

@test_nowarn sequentialcovering(X, y_clabel; max_rulebase_length=1, suppress_parity_warning=true)
@test_logs (:warn,"Parity encountered in bestguess! counts (149 elements):" *
            " Dict(2 => 50, 3 => 50, 1 => 49), argmax: 2, max: 50 (sum = 149)"
    ) sequentialcovering(X, y_clabel; max_rulebase_length=1)

@test_nowarn sequentialcovering(X, y_clabel; max_rulebase_length=3, suppress_parity_warning=true)
dl = sequentialcovering(X, y_clabel; max_rulebase_length=3, suppress_parity_warning=true)
@test length(rulebase(dl)) <= 3

@test_nowarn sequentialcovering(X, y_clabel; max_rulebase_length=5, suppress_parity_warning=true)
dl = sequentialcovering(X, y_clabel; max_rulebase_length=5, suppress_parity_warning=true)
@test length(rulebase(dl)) <= 5

@test_nowarn sequentialcovering(X, y_clabel; max_rulebase_length=1000, suppress_parity_warning=true)

############################################################################################
############################## beam_width ##################################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel; beam_width=1)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=3)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=5)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=25)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=5))
#=  Beam = 0 =#
@test_throws AssertionError sequentialcovering(X, y_clabel; beam_width=0)
@test_throws AssertionError sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=0))
#= Mi assicuro che il parametro venga sovrascrito =#
@test_throws AssertionError sequentialcovering(X, y_clabel; beam_width=0, searchmethod=BeamSearch(; beam_width=5))

############################################################################################
############################## quality_evaluator ###########################################
############################################################################################

bs5_ent = BeamSearch(; beam_width=5, quality_evaluator=entropy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs5_ent)

bs_entropy = BeamSearch(; quality_evaluator=entropy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs_entropy)

bs_laplace = BeamSearch(; quality_evaluator=laplace_accuracy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs_laplace)
@test_nowarn sequentialcovering(X, y_intger; searchmethod=bs_laplace)
@test_nowarn sequentialcovering(X, y_catgcl; searchmethod=bs_laplace)
@test_nowarn sequentialcovering(X, y_clabel; quality_evaluator=laplace_accuracy)

@test_nowarn sequentialcovering(X, y_clabel; beam_width=1, quality_evaluator=laplace_accuracy)

############################################################################################
############################## beam_width ##################################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel; beam_width=1)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=3)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=5)
@test_nowarn sequentialcovering(X, y_clabel; beam_width=25)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=5))
#=  Beam = 0 =#
@test_throws AssertionError sequentialcovering(X, y_clabel; beam_width=0)
@test_throws AssertionError sequentialcovering(X, y_clabel; searchmethod=BeamSearch(; beam_width=0))
#= Mi assicuro che il parametro venga sovrascrito =#
@test_throws AssertionError sequentialcovering(X, y_clabel; beam_width=0, searchmethod=BeamSearch(; beam_width=5))

############################################################################################
############################## quality_evaluator ###########################################
############################################################################################

bs5_ent = BeamSearch(; beam_width=5, quality_evaluator=entropy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs5_ent)

bs_entropy = BeamSearch(; quality_evaluator=entropy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs_entropy)

bs_laplace = BeamSearch(; quality_evaluator=laplace_accuracy)
@test_nowarn sequentialcovering(X, y_clabel; searchmethod=bs_laplace)
@test_nowarn sequentialcovering(X, y_intger; searchmethod=bs_laplace)
############################################################################################
############################## quality_evaluator + weights #################################
############################################################################################

bs_laplace = BeamSearch(; quality_evaluator=laplace_accuracy)
@test_nowarn sequentialcovering(X, y_clabel, w; searchmethod=bs_laplace)


############################################################################################
############################## truerfirst #################################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel; truerfirst=true)
@test_nowarn sequentialcovering(X, y_clabel; truerfirst=true, beam_width=1)
@test_nowarn sequentialcovering(X, y_clabel; truerfirst=true, quality_evaluator=laplace_accuracy)
@test_nowarn sequentialcovering(X, y_clabel; truerfirst=true, max_rulebase_length=2, suppress_parity_warning = true)

############################################################################################
############################## discretizedomain ############################################
############################################################################################

@test_nowarn sequentialcovering(X, y_clabel; discretizedomain=true)
