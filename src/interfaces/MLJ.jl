module MLJInterface

export ExtendedSequentialCovering
export OrderedCN2Learner
# export BeamSearch, RandSearch

using ModalDecisionLists
using ModalDecisionLists: LossFunctions

# import ModalDecisionLists: SearchMethod, BeamSearch, RandSearch
# import ModalDecisionLists: sequentialcovering

using SoleData
import SoleBase: CLabel
import SoleModels: apply

import MLJModelInterface
using Parameters
using StatsBase

const MMI = MLJModelInterface
const MDL = ModalDecisionLists

const _package_url = "https://github.com/aclai-lab/$(MDL).jl"

abstract type CoveringStrategy <: MLJModelInterface.Deterministic end

############################################################################################
############################ ExtendedSequentialCovering #####################################
############################################################################################
"""

        ExtendedSequentialCovering

A model type for constructing a sequential covering classifier, based on ModalDecisionLists.jl,
and implementing the MLJ model interface.

Do `model = ExtendedSequentialCovering()` to contruct an instance with default hyper-parameters. Provide
keyword arguments to override hyper-parameter defaults, as in `ExtendedSequentialCovering(searchmethod=...)`


## Training data

In MLJ or MLJBase, bind an instance model to data with

`mach = machine(model, X, y)`

where

- X: ant table of input feature (eg, a DataFrame); TODO continue here....
- y: the target, which can be any AbstractVector whose element scitype is <:MultiClass; check the scitype with scitype(y)

Train the machine with  fit!(mach, rows=...).

## Hyperparameters

* `searchmethod::SearchMethod=BeamSearch()`: The search method for finding single rules (see [`SearchMethod`](@ref)).
* `max_rulebase_length::Union{Nothing,Integer}=nothing` is the maximum length of the rulebase.
* `min_rule_coverage::Integer=1`: constrains the minimum number of instances covered by each rule.
* `suppress_parity_warning::Bool=false`: if `true`, suppresses parity warnings.

## Fitted Parameters

The fileds of fitted_params(mach) are:

* `fitresult`: A `DecisionList` object.

## Operations

* `predict(mach, Xnew)`: Return a vector of predictions for each row of Xnew.

See also
[`SearchMethod`](@ref),
"""
mutable struct ExtendedSequentialCovering <: CoveringStrategy
    searchmethod::SearchMethod
    # shared parameters
    loss_function::Function
    discretizedomain::Bool
    max_infogain_ratio::Real
    significance_alpha::Union{Real,Nothing}
    min_rule_coverage::Union{Nothing,Integer}
    max_rulebase_length::Union{Nothing,Integer}
    suppress_parity_warning::Bool
end

function MMI.clean!(model::ExtendedSequentialCovering)
    warning = ""
    if !isnothing(model.max_rulebase_length) && model.max_rulebase_length < 1
        warning *= "Need max_rulebase_length ≥ 1. Resetting max_rulebase_length = nothing. "
        #
        model.max_rulebase_length = nothing
    end
    return warning
end

# Keyword constructor
function ExtendedSequentialCovering(;
    searchmethod::SearchMethod=BeamSearch(),
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    # shared parameters
    loss_function::Function=LossFunctions.entropy,
    discretizedomain::Bool=false,
    max_infogain_ratio::Real=1.0,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Integer=1,

    suppress_parity_warning::Bool=false,
    kwargs...
)
    searchmethod = reconstruct(searchmethod,  kwargs)
    model =  ExtendedSequentialCovering(searchmethod,
        loss_function, discretizedomain, max_infogain_ratio, significance_alpha, min_rule_coverage,
        max_rulebase_length, suppress_parity_warning
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end

function MMI.predict(m::CoveringStrategy, fitresult, Xnew)
    yhat = apply(fitresult.model, PropositionalLogiset(Xnew))
    return yhat
end

############################################################################################
############################ OrderedCN2Learner #############################################
############################################################################################

mutable struct OrderedCN2Learner <: CoveringStrategy

    beam_width::Integer
    loss_function::Function
    discretizedomain::Bool
    max_infogain_ratio::Union{Real,Nothing}
    significance_alpha::Union{Real,Nothing}
    # SequentialCovering
    min_rule_coverage::Integer
    max_rule_length::Union{Real,Nothing}
    max_rulebase_length::Union{Nothing,Integer}
end

function MMI.clean!(model::OrderedCN2Learner)
    warning = ""
    if !isnothing(model.max_rulebase_length) && model.max_rulebase_length < 1
        warning *= "Need max_rulebase_length ≥ 1. Resetting max_rulebase_length = nothing. "
        #
        model.max_rulebase_length = nothing
    end
    return warning
end

# Keyword constructor
function OrderedCN2Learner(;
    beam_width::Integer = 3,
    loss_function::Function = ModalDecisionLists.LossFunctions.entropy,
    discretizedomain::Bool = false,
    max_infogain_ratio::Union{Real,Nothing} = nothing,
    significance_alpha::Union{Real,Nothing} = nothing,
    # SequentialCovering
    min_rule_coverage::Integer = 1,
    max_rule_length::Union{Nothing,Integer} = nothing,
    max_rulebase_length::Union{Nothing,Integer} = nothing,
)
    model = OrderedCN2Learner(beam_width,
        loss_function, discretizedomain,
        max_infogain_ratio, significance_alpha,
        min_rule_coverage, max_rule_length, max_rulebase_length,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end

################ Fit (General for all CoveringStrategy ) ###################################
############################################################################################

function MMI.fit(m::CoveringStrategy, verbosity::Integer, X, y)

    # TODO use wrapdataset...?
    X_pl = PropositionalLogiset(X)
    y_cl = Vector{CLabel}(y)

    model = begin
        if m isa ExtendedSequentialCovering
            sequentialcovering(X_pl, y_cl;
                        searchmethod              = m.searchmethod,
                        loss_function             = m.loss_function,
                        discretizedomain          = m.discretizedomain,
                        max_infogain_ratio             = m.max_infogain_ratio,
                        significance_alpha        = m.significance_alpha,
                        min_rule_coverage         = m.min_rule_coverage,
                        max_rulebase_length       = m.max_rulebase_length,
                        suppress_parity_warning   = m.suppress_parity_warning
        )
        # elseif m isa OrderedCN2Learner
            # searchmethod = BeamSearch( conjuncts_search_method = AtomSearch(),
            #     beam_width          = m.beam_width,
            #     loss_function       = m.loss_function,
            #     discretizedomain    = m.discretizedomain,
            #     max_infogain_ratio    = m.max_infogain_ratio,
            #     significance_alpha  = m.significance_alpha,
            # )
            # sequentialcovering(X_pl, y_cl;
            #             searchmethod,
            #             m.min_rule_coverage,
            #             m.max_rule_length,
            #             m.max_rulebase_length
            # )
        else
            error("unexpected model type $(typeof(model))")
        end
    end
    if verbosity == 1
        println(model)
    end
    fitresult = (
        model = model,
    )
    report = (
        model = model,
    )
    cache = nothing

    return fitresult, cache, report
end

MMI.metadata_pkg.(
    (
        OrderedCN2Learner,
        ExtendedSequentialCovering,
    ),
    name = "$(MDL)",
    package_uuid = "dbece2fb-9d58-4710-9902-4ec759308ae8",
    package_url = _package_url,
    is_pure_julia = true,
    is_wrapper=false,
    package_license = "MIT",
)

end # module
