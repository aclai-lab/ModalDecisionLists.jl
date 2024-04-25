module MLJInterface

export SequentialCoveringLearner
# export BeamSearch, RandSearch

# using ModalDecisionTrees.MLJInterface: wrapdataset
using ModalDecisionLists
import ModalDecisionLists: SearchMethod, BeamSearch, RandSearch
import ModalDecisionLists:  sequentialcovering

import SoleData: PropositionalLogiset
import SoleBase: CLabel
import SoleModels: apply

import MLJModelInterface
using Parameters

const MMI = MLJModelInterface
const MDL = ModalDecisionLists

const _package_url = "https://github.com/aclai-lab/$(MDL).jl"

abstract type CoveringStrategy <: MLJModelInterface.Deterministic end

############################################################################################
############################ SequentialCoveringLearner #####################################
############################################################################################

mutable struct SequentialCoveringLearner <: CoveringStrategy
    searchmethod::SearchMethod
    max_rulebase_length::Union{Nothing,Integer}
    suppress_parity_warning::Bool
end

function MMI.clean!(model::SequentialCoveringLearner)
    warning = ""
    if !isnothing(model.max_rulebase_length) && model.max_rulebase_length < 1
        warning *= "Need max_rulebase_length ≥ 1. Resetting max_rulebase_length = nothing. "
        #
        model.max_rulebase_length = nothing
    end
    return warning
end

# Keyword constructor
function SequentialCoveringLearner(;
    searchmethod::SearchMethod=BeamSearch(),
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    suppress_parity_warning::Bool=false,
    kwargs...
)
    searchmethod = reconstruct(searchmethod,  kwargs)
    model =  SequentialCoveringLearner(
        searchmethod,
        max_rulebase_length,
        suppress_parity_warning
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end

function MMI.fit(m::CoveringStrategy, verbosity::Integer, X, y)

    # TODO wrapdataset
    X_pl = PropositionalLogiset(X)
    y_cl = Vector{CLabel}(y)

    model = begin
        if m isa SequentialCoveringLearner
            sequentialcovering(
                        X_pl, y_cl;
                        m.searchmethod,
                        m.max_rulebase_length,
                        m.suppress_parity_warning)
        else
            error("unexpected model type $(typeof(model))")
        end
    end
    fitresult = (
        model = model,
    )
    if verbosity > 1
        println(model)
    end
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(m::SequentialCoveringLearner, fitresult, Xnew)
    yhat = apply(fitresult.model, PropositionalLogiset(Xnew))
    return yhat
end

MMI.metadata_pkg.(
    (
        SequentialCoveringLearner,
    ),
    name = "$(MDL)",
    package_uuid = "dbece2fb-9d58-4710-9902-4ec759308ae8",
    package_url = _package_url,
    is_pure_julia = true,
    is_wrapper=false,
    package_license = "MIT",
)

end # module
