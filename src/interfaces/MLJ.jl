module MLJInterface

export ExtendedSequentialCovering, OrderedCN2Learner
export DecisionListClassifier, RandomDecisionListClassifier
# export BeamSearch, RandSearch

using ModalDecisionLists
using ModalDecisionLists: LossFunctions
using ModalDecisionLists: Metrics
using ModalDecisionLists: AbstractGenerator, AtomGenerator

# import ModalDecisionLists: SearchMethod, BeamSearch, RandSearch
# import ModalDecisionLists: sequentialcovering

using SoleLogics: AbstractAlphabet

using SoleData
import SoleBase: CLabel
import SoleModels: apply

import MLJModelInterface
using Parameters
using StatsBase
using Random

const MMI = MLJModelInterface
const MDL = ModalDecisionLists

const _package_url = "https://github.com/aclai-lab/$(MDL).jl"

abstract type CoveringStrategy <: MMI.Deterministic end

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
* `max_rulebase_length::Union{Nothing,Int}=nothing` is the maximum length of the rulebase.
* `min_rule_coverage::Int=1`: constrains the minimum number of instances covered by each rule.
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
    loss_function::LossFunctions.AbstractLossFunction
    discretizedomain::Bool
    max_infogain_ratio::Real
    significance_alpha::Union{Real,Nothing}
    min_rule_coverage::Union{Nothing,Int}
    max_rulebase_length::Union{Nothing,Int}
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
    max_rulebase_length::Union{Nothing,Int}=nothing,
    # shared parameters
    loss_function::LossFunctions.AbstractLossFunction = LossFunctions.Entropy(),
    discretizedomain::Bool=false,
    max_infogain_ratio::Real=1.0,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Int=1,

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

    beam_width::Int
    loss_function::LossFunctions.SymmetricLoss
    discretizedomain::Bool
    max_infogain_ratio::Union{Real,Nothing}
    significance_alpha::Union{Real,Nothing}
    # SequentialCovering
    min_rule_coverage::Int
    max_rule_length::Union{Real,Nothing}
    max_rulebase_length::Union{Nothing,Int}
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
    beam_width::Int = 3,
    loss_function::LossFunctions.AbstractLossFunction = LossFunctions.Entropy(),
    discretizedomain::Bool = false,
    max_infogain_ratio::Union{Real,Nothing} = nothing,
    significance_alpha::Union{Real,Nothing} = nothing,
    # SequentialCovering
    min_rule_coverage::Int = 1,
    max_rule_length::Union{Nothing,Int} = nothing,
    max_rulebase_length::Union{Nothing,Int} = nothing,
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

function MMI.fit(m::CoveringStrategy, verbosity::Int, X, y)

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

# ---------------------------------------------------------------------------- #
#                          decision list classifier                            #
# ---------------------------------------------------------------------------- #
mutable struct DecisionListClassifier <: CoveringStrategy
    searchmethod::SearchMethod 
    tdl_threshold::Int
    split_ratio::Real
    loss_function::LossFunctions.AsymmetricLoss
    max_infogain_ratio::Union{Nothing,Real}
    default_alphabet::Union{Nothing,AbstractAlphabet}
    discretizedomain::Bool
    significance_alpha::Union{Real,Nothing}
    min_rule_coverage::Int
    max_rule_length::Union{Nothing,Int}
    max_rulebase_length::Union{Nothing,Int}
    conjuncts_generation_method::AbstractGenerator
    beam_width::Int
    rng::AbstractRNG
    suppress_parity_warning::Bool
end

function DecisionListClassifier(;
    searchmethod::SearchMethod=BeamSearch(), 
    tdl_threshold::Int=64,
    split_ratio::Real=0.7, 
    loss_function::LossFunctions.AsymmetricLoss=LossFunctions.LaplaceAccuracy(),
    max_infogain_ratio::Union{Nothing,Real}=nothing,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    discretizedomain::Bool=false,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Int=1, 
    max_rule_length::Union{Nothing,Int}=nothing,
    max_rulebase_length::Union{Nothing,Int}=nothing,
    # BeamSearch
    conjuncts_generation_method::AbstractGenerator=AtomGenerator(),
    beam_width::Int=3,
    # utils
    rng::AbstractRNG=TaskLocalRNG(),
    suppress_parity_warning::Bool=false,
)
    model = DecisionListClassifier(
        searchmethod, 
        tdl_threshold,
        split_ratio, 
        loss_function,
        max_infogain_ratio,
        default_alphabet,
        discretizedomain,
        significance_alpha,
        min_rule_coverage, 
        max_rule_length,
        max_rulebase_length,
        conjuncts_generation_method,
        beam_width,
        rng,
        suppress_parity_warning,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::DecisionListClassifier)
    warning = ""
    if !isnothing(model.max_rulebase_length) && model.max_rulebase_length < 1
        warning *= "Need max_rulebase_length ≥ 1. " *
            "Resetting max_rulebase_length = nothing."
        model.max_rulebase_length = nothing
    end
    return warning
end

function MMI.fit(m::DecisionListClassifier, verbosity::Int, X, y)
    featurenames = propertynames(X)
    logiset = scalarlogiset(X; featurenames, allow_propositional=true)

    model = begin
        irepstar(
            logiset,
            y;
            featurenames,
            searchmethod=m.searchmethod,
            tdl_threshold=m.tdl_threshold,
            split_ratio=m.split_ratio,
            loss_function=m.loss_function,
            max_infogain_ratio=m.max_infogain_ratio,
            default_alphabet=m.default_alphabet,
            discretizedomain=m.discretizedomain,
            significance_alpha=m.significance_alpha,
            min_rule_coverage=m.min_rule_coverage,
            max_rule_length=m.max_rule_length,
            max_rulebase_length=m.max_rulebase_length,
            conjuncts_generation_method=m.conjuncts_generation_method,
            beam_width=m.beam_width,
            rng=m.rng,
            suppress_parity_warning=m.suppress_parity_warning
        )
    end

    verbosity == 1 && println(model)

    fitresult = (; model)
    report = (; model)
    cache = nothing

    return fitresult, cache, report
end

# ---------------------------------------------------------------------------- #
#                      random decision tree classifier                         #
# ---------------------------------------------------------------------------- #
mutable struct RandomDecisionListClassifier <: CoveringStrategy
    num_models::Int
    use_bootstrapping::Bool
    samples_ratio_per_model::Real
    n_subfeatures_per_model::Union{Nothing,Int}
    aggregation_function::Union{Nothing,Base.Callable}
    searchmethod::SearchMethod 
    tdl_threshold::Int
    split_ratio::Real
    loss_function::LossFunctions.AsymmetricLoss
    max_infogain_ratio::Union{Nothing,Real}
    default_alphabet::Union{Nothing,AbstractAlphabet}
    discretizedomain::Bool
    significance_alpha::Union{Real,Nothing}
    min_rule_coverage::Int
    max_rule_length::Union{Nothing,Int}
    max_rulebase_length::Union{Nothing,Int}
    conjuncts_generation_method::AbstractGenerator
    beam_width::Int
    rng::AbstractRNG
    suppress_parity_warning::Bool
end

function RandomDecisionListClassifier(;
    # ensemble
    num_models::Int=50,
    use_bootstrapping::Bool=true,
    samples_ratio_per_model::Real=1.0,
    n_subfeatures_per_model::Union{Nothing,Int}=nothing,
    aggregation_function::Union{Nothing,Base.Callable}=nothing,
    #irepstar
    searchmethod::SearchMethod=BeamSearch(), 
    tdl_threshold::Int=64,
    split_ratio::Real=0.7, 
    loss_function::LossFunctions.AsymmetricLoss=LossFunctions.LaplaceAccuracy(),
    max_infogain_ratio::Union{Nothing,Real}=nothing,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    discretizedomain::Bool=false,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Int=1, 
    max_rule_length::Union{Nothing,Int}=nothing,
    max_rulebase_length::Union{Nothing,Int}=nothing,
    # BeamSearch
    conjuncts_generation_method::AbstractGenerator=AtomGenerator(),
    beam_width::Int=3,
    # utils
    rng::AbstractRNG=TaskLocalRNG(),
    suppress_parity_warning::Bool=false,
)
    model = RandomDecisionListClassifier(
        num_models,
        use_bootstrapping,
        samples_ratio_per_model,
        n_subfeatures_per_model,
        aggregation_function,
        searchmethod, 
        tdl_threshold,
        split_ratio, 
        loss_function,
        max_infogain_ratio,
        default_alphabet,
        discretizedomain,
        significance_alpha,
        min_rule_coverage, 
        max_rule_length,
        max_rulebase_length,
        conjuncts_generation_method,
        beam_width,
        rng,
        suppress_parity_warning,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::RandomDecisionListClassifier)
    warning = ""
    if !isnothing(model.max_rulebase_length) && model.max_rulebase_length < 1
        warning *= "Need max_rulebase_length ≥ 1. " *
            "Resetting max_rulebase_length = nothing."
        model.max_rulebase_length = nothing
    end
    return warning
end

function MMI.fit(m::RandomDecisionListClassifier, verbosity::Int, X, y)
    featurenames = propertynames(X)
    logiset = scalarlogiset(X; featurenames, allow_propositional=true)

    model = begin
        build_ensemble(
            logiset,
            y,
            m.num_models;
            use_bootstrapping=m.use_bootstrapping,
            samples_ratio_per_model=m.samples_ratio_per_model,
            n_subfeatures_per_model=m.n_subfeatures_per_model,
            aggregation_function=m.aggregation_function,
            model_wrapper=sequentialcovering,
            rng=m.rng,

            max_rulebase_length=15
            # irepstar kwargs
            # searchmethod=m.searchmethod,
            # tdl_threshold=m.tdl_threshold,
            # split_ratio=m.split_ratio,
            # loss_function=m.loss_function,
            # max_infogain_ratio=m.max_infogain_ratio,
            # default_alphabet=m.default_alphabet,
            # discretizedomain=m.discretizedomain,
            # significance_alpha=m.significance_alpha,
            # min_rule_coverage=m.min_rule_coverage,
            # max_rule_length=m.max_rule_length,
            # max_rulebase_length=m.max_rulebase_length,
            # conjuncts_generation_method=m.conjuncts_generation_method,
            # beam_width=m.beam_width,
            # suppress_parity_warning=m.suppress_parity_warning
        )
    end

    verbosity == 1 && println(model)

    fitresult = (; model)
    report = (; model)
    cache = nothing

    return fitresult, cache, report
end

# ---------------------------------------------------------------------------- #
#                                   metadata                                   #
# ---------------------------------------------------------------------------- #
MMI.prediction_type(::Type{<:DecisionListClassifier}) = :probabilistic
MMI.prediction_type(::Type{<:RandomDecisionListClassifier}) = :probabilistic

MMI.metadata_pkg.(
    (
        OrderedCN2Learner,
        ExtendedSequentialCovering,
        DecisionListClassifier,
        RandomDecisionListClassifier,
    ),
    name = "$(MDL)",
    package_uuid = "dbece2fb-9d58-4710-9902-4ec759308ae8",
    package_url = _package_url,
    is_pure_julia = true,
    is_wrapper=false,
    package_license = "MIT",
)

end
