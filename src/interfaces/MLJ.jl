module MLJInterface

export ExtendedSequentialCovering, OrderedCN2Learner
export DecisionListClassifier, BaggedEnsembleClassifier
export RipperListClassifier, RandomDecisionListEnsembleClassifier

using ModalDecisionLists
using ModalDecisionLists: LossFunctions
using ModalDecisionLists: Metrics
using ModalDecisionLists: AbstractGenerator, AtomGenerator


using SoleLogics: AbstractAlphabet

using SoleData
import SoleBase: CLabel
import SoleModels: apply

import MLJModelInterface
using Parameters
using StatsBase
using Random
using CategoricalArrays: levels, isordered, unwrap

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
    min_rule_coverage::Int=3,

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
    raw_preds = apply(fitresult.model, PropositionalLogiset(Xnew))
    unwrapped_preds = unwrap.(raw_preds)
    return MMI.categorical(unwrapped_preds, levels=levels(fitresult.target_pool), ordered=isordered(fitresult.target_pool))
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
                        max_infogain_ratio        = m.max_infogain_ratio,
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

    target_pool = MMI.categorical(y)        # extract y levels
    fitresult = (; model, target_pool)
    
    report = (
        model = model,
    )
    cache = nothing

    return fitresult, cache, report
end

# ---------------------------------------------------------------------------- #
#                          decision list classifier (irep*)                    #
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
    invert_class_orders::Bool
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
    min_rule_coverage::Int=2, 
    max_rule_length::Union{Nothing,Int}=nothing,
    max_rulebase_length::Union{Nothing,Int}=nothing,
    # BeamSearch
    conjuncts_generation_method::AbstractGenerator=AtomGenerator(),
    beam_width::Int=3,
    # utils
    rng::AbstractRNG=TaskLocalRNG(),
    suppress_parity_warning::Bool=false,
    invert_class_orders::Bool = false
)
    searchmethod.beam_width = beam_width

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
        invert_class_orders
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
    logiset = PropositionalLogiset(X)

    model = begin
        irepstar(
            logiset,
            y;
            featurenames,
            searchmethod                    = m.searchmethod,
            tdl_threshold                   = m.tdl_threshold,
            split_ratio                     = m.split_ratio,
            loss_function                   = m.loss_function,
            max_infogain_ratio              = m.max_infogain_ratio,
            default_alphabet                = m.default_alphabet,
            discretizedomain                = m.discretizedomain,
            significance_alpha              = m.significance_alpha,
            min_rule_coverage               = m.min_rule_coverage,
            max_rule_length                 = m.max_rule_length,
            max_rulebase_length             = m.max_rulebase_length,
            conjuncts_generation_method     = m.conjuncts_generation_method,
            beam_width                      = m.beam_width,
            rng                             = m.rng,
            suppress_parity_warning         = m.suppress_parity_warning,
            invert_class_orders             = m.invert_class_orders
        )
    end

    verbosity == 1 && println(model)

    target_pool = MMI.categorical(y)        # extract y levels
    fitresult = (; model, target_pool)
    report = (
        model = model,
    )
    cache = nothing

    return fitresult, cache, report
end



# ---------------------------------------------------------------------------- #
#                          decision list classifier (ripper)                   #
# ---------------------------------------------------------------------------- #
mutable struct RipperListClassifier <: CoveringStrategy
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
    max_k::Int
    
    # AtomGenerator
    conjuncts_generation_method::AbstractGenerator
    
    # BeamSearch
    beam_width::Int
    rng::AbstractRNG
    suppress_parity_warning::Bool
    invert_class_orders::Bool
end

function RipperListClassifier(;
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
    max_k::Integer = 2,
    invert_class_orders::Bool = false,

    # BeamSearch
    conjuncts_generation_method::AbstractGenerator=AtomGenerator(),
    beam_width::Int=3,
    # utils
    rng::AbstractRNG=TaskLocalRNG(),
    suppress_parity_warning::Bool=false
)
    searchmethod.beam_width = beam_width

    model = RipperListClassifier(
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
        max_k,
        conjuncts_generation_method,
        beam_width,
        rng,
        suppress_parity_warning,
        invert_class_orders
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::RipperListClassifier)
    warning = ""
    if !isnothing(model.max_rulebase_length) && model.max_rulebase_length < 1
        warning *= "Need max_rulebase_length ≥ 1. " *
            "Resetting max_rulebase_length = nothing."
        model.max_rulebase_length = nothing
    end
    return warning
end

function MMI.fit(m::RipperListClassifier, verbosity::Int, X, y)
    featurenames = propertynames(X)
    logiset = PropositionalLogiset(X)

    model = begin
        ripperk(
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
            max_k = m.max_k,
            conjuncts_generation_method=m.conjuncts_generation_method,
            beam_width=m.beam_width,
            rng=m.rng,
            suppress_parity_warning=m.suppress_parity_warning,
            invert_class_orders=m.invert_class_orders
        )
    end

    verbosity == 1 && println(model)

    target_pool = MMI.categorical(y)        # extract y levels
    fitresult = (; model, target_pool)

    report = (
        model = model,
    )
    cache = nothing

    return fitresult, cache, report
end









# ---------------------------------------------------------------------------- #
#                      bagging ensemble classifier                         #
# ---------------------------------------------------------------------------- #
mutable struct BaggedEnsembleClassifier <: CoveringStrategy
    num_models::Int
    use_bootstrapping::Bool
    samples_ratio_per_model::Real
    n_subfeatures_per_model::Union{Nothing,Int}
    aggregation_function::Union{Nothing,Base.Callable}
    base_model::CoveringStrategy
    rng::AbstractRNG
end

function BaggedEnsembleClassifier(;
    # ensemble
    num_models::Int=50,
    use_bootstrapping::Bool=true,
    samples_ratio_per_model::Real=1.0,
    n_subfeatures_per_model::Union{Nothing,Int}=nothing,
    aggregation_function::Union{Nothing,Base.Callable}=nothing,
    base_model::CoveringStrategy = DecisionListClassifier(),
    rng::AbstractRNG = TaskLocalRNG(),
)
    model = BaggedEnsembleClassifier(
        num_models,
        use_bootstrapping,
        samples_ratio_per_model,
        n_subfeatures_per_model,
        aggregation_function,
        base_model,
        rng,
    )

    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::BaggedEnsembleClassifier)
    warning = ""
    if model.num_models <= 0
        warning *= "Need num_models ≥ 1. " *
            "Resetting num_models = 1."
        model.max_rulebase_length = 1
    end
    if isa(model.base_model, BaggedEnsembleClassifier) || isa(model.base_model, RandomDecisionListEnsembleClassifier)
        error("Cannot create a RandomDecisionListEnsembleClassifier with a base model type: $(typeof(model.base_model))")
    end 
    return warning
end

function MMI.fit(m::BaggedEnsembleClassifier, verbosity::Int, X, y)
    featurenames = propertynames(X)
    # logiset = scalarlogiset(X; featurenames, allow_propositional=true)
    logiset = PropositionalLogiset(X)

    model_wrappers = Dict(
        ExtendedSequentialCovering      => sequentialcovering,
        DecisionListClassifier          => irepstar,
        RipperListClassifier            => ripperk,
    )
    model_wrapper = model_wrappers[typeof(m.base_model)]

    base_model_kwargs = Dict(p => getproperty(m.base_model, p) for p in propertynames(m.base_model))

    model = begin
        build_ensemble(
            logiset,
            y,
            m.num_models;
            featurenames,
            use_bootstrapping           = m.use_bootstrapping,
            samples_ratio_per_model     = m.samples_ratio_per_model,
            n_subfeatures_per_model     = m.n_subfeatures_per_model,
            aggregation_function        = m.aggregation_function,
            model_wrapper               = model_wrapper,
            rng                         = m.rng,

            base_model_kwargs...
        )
    end

    verbosity == 1 && println(model)

    target_pool = MMI.categorical(y)        # extract y levels
    fitresult = (; model, target_pool)
    report = (; model)
    cache = nothing

    return fitresult, cache, report
end

function MMI.predict(m::BaggedEnsembleClassifier, fitresult, Xnew)
    raw_preds = apply(fitresult.model, PropositionalLogiset(Xnew); use_multithreads=false, suppress_parity_warning=true)
    unwrapped_preds = unwrap.(raw_preds)
    return MMI.categorical(unwrapped_preds, levels=levels(fitresult.target_pool), ordered=isordered(fitresult.target_pool))
end






# ---------------------------------------------------------------------------- #
#                      RandomDecisionListEnsembleClassifier                    #
# ---------------------------------------------------------------------------- #
mutable struct RandomDecisionListEnsembleClassifier <: MMI.Deterministic
    num_models::Int
    use_bootstrapping::Bool
    samples_ratio_per_model::Real
    n_subfeatures_per_model::Union{Nothing,Int}
    beta::Real
    base_model::CoveringStrategy
    prop_features_ratio::Real
    num_features_per_proposition::Union{Integer, Nothing}
    rng::AbstractRNG
end

function RandomDecisionListEnsembleClassifier(;
    num_models::Int = 10,
    use_bootstrapping::Bool = true,
    samples_ratio_per_model::Real = 1.0,
    n_subfeatures_per_model::Union{Nothing,Int} = nothing,
    beta::Real = 1.0,
    base_model::CoveringStrategy = DecisionListClassifier(),
    prop_features_ratio::Real = 1.0,
    num_features_per_proposition::Union{Integer, Nothing} = nothing,
    rng::AbstractRNG = TaskLocalRNG(),
)
    model = RandomDecisionListEnsembleClassifier(
        num_models,
        use_bootstrapping,
        samples_ratio_per_model,
        n_subfeatures_per_model,
        beta,
        base_model,
        prop_features_ratio,
        num_features_per_proposition,
        rng,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::RandomDecisionListEnsembleClassifier)
    warning = ""
    if model.samples_ratio_per_model <= 0.0
        warning *= "Need samples_ratio_per_model > 0. " *
            "Resetting samples_ratio_per_model = 1.0."
        model.samples_ratio_per_model = 1.0
    end
    if model.beta < 0.0
        warning *= "Need alpha ≥ 0. Resetting alpha = 0.0."
        model.beta = 0.0
    end
    if isa(model.base_model, BaggedEnsembleClassifier) || isa(model.base_model, RandomDecisionListEnsembleClassifier)
        error("Cannot create a RandomDecisionListEnsembleClassifier with a base model type: $(typeof(model.base_model))")
    end 
    return warning
end

function MMI.fit(m::RandomDecisionListEnsembleClassifier, verbosity::Int, X, y)
    featurenames = propertynames(X)
    logiset = PropositionalLogiset(X)

    model_wrappers = Dict(
        ExtendedSequentialCovering       => sequentialcovering,
        DecisionListClassifier           => irepstar,
        RipperListClassifier             => ripperk,
    )
    model_wrapper = model_wrappers[typeof(m.base_model)]

    num_features_per_proposition = if isnothing(m.num_features_per_proposition)
        n_sub = isnothing(m.n_subfeatures_per_model) ? length(featurenames) : m.n_subfeatures_per_model
        max(1, round(Int, m.prop_features_ratio * n_sub))
    else
        m.num_features_per_proposition
    end

    feature_selection_strategy = WeightedRandomFeatureSelector(m.beta, num_features_per_proposition)

    base_model_kwargs = Dict(p => getproperty(m.base_model, p) for p in propertynames(m.base_model))


    model = build_random_lists(
        logiset,
        y,
        m.num_models;
        featurenames,
        use_bootstrapping           = m.use_bootstrapping,
        samples_ratio_per_model     = m.samples_ratio_per_model,
        n_subfeatures_per_model     = m.n_subfeatures_per_model,
        model_wrapper               = model_wrapper,
        feature_selection_strategy,
        rng                         = m.rng,

        base_model_kwargs...
    )

    verbosity == 1 && println(model)

    target_pool = MMI.categorical(y)        # extract y levels
    fitresult = (; model, target_pool)
    report    = (; model)
    cache     = nothing

    return fitresult, cache, report
end

function MMI.predict(m::RandomDecisionListEnsembleClassifier, fitresult, Xnew)
    raw_preds = ModalDecisionLists.apply_rdl(fitresult.model, PropositionalLogiset(Xnew))
    unwrapped_preds = unwrap.(raw_preds)
    return MMI.categorical(unwrapped_preds, levels=levels(fitresult.target_pool), ordered=isordered(fitresult.target_pool))
end




# ---------------------------------------------------------------------------- #
#                                   metadata                                   #
# ---------------------------------------------------------------------------- #
# MMI.prediction_type(::Type{<:DecisionListClassifier}) = :probabilistic
# MMI.prediction_type(::Type{<:RandomDecisionListClassifier}) = :probabilistic

MMI.metadata_pkg.(
    (
        OrderedCN2Learner,
        ExtendedSequentialCovering,
        DecisionListClassifier,
        RipperListClassifier,
        BaggedEnsembleClassifier,
        RandomDecisionListEnsembleClassifier
    ),
    name = "$(MDL)",
    package_uuid = "dbece2fb-9d58-4710-9902-4ec759308ae8",
    package_url = _package_url,
    is_pure_julia = true,
    is_wrapper=false,
    package_license = "MIT",
)

end
