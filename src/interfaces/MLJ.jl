module MLJInterface

export SequentialCoveringLearner
export BeamSearch, RandSearch

using ModalDecisionLists
import ModalDecisionLists: SearchMethod, BeamSearch, RandSearch
import ModalDecisionLists:  sequentialcovering

import SoleData: PropositionalLogiset
import SoleBase: CLabel
import SoleModels: apply
import MLJBase

import MLJModelInterface
using Parameters

const MMI = MLJModelInterface
const MDL = ModalDecisionLists

const _package_url = "https://github.com/aclai-lab/$(MDL).jl"



############################################################################################
############################ SequentialCoveringLearner #####################################
############################################################################################

mutable struct SequentialCoveringLearner <: MLJModelInterface.Deterministic
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

function MMI.fit(model::SequentialCoveringLearner, verbosity, X, y)
    fitresult = sequentialcovering(
                        PropositionalLogiset(X),
                        Vector{CLabel}(y);
                        model.searchmethod,
                        model.max_rulebase_length,
                        model.suppress_parity_warning)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::SequentialCoveringLearner, fitresult, Xnew)
    yhat = apply(fitresult, PropositionalLogiset(Xnew))
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
#=
[Easy test] -- Copiare ed incollare nel terminale:

include("ModalDecisionLists/src/mlj.jl")
seqcovering_model = SequentialCoveringLearner()

include("ModalDecisionLists/test/cn2-accuracy.jl")
learned_list = machine(seqcovering_model, iris_df, CategoricalValue.(y_iris))
fit!(learned_list)
MLJBase.predict(learned_list, iris_df)
=#
