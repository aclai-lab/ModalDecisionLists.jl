import ModalDecisionLists: SearchMethod, BeamSearch, sequentialcovering
import SoleData: PropositionalLogiset
import SoleBase: CLabel
import SoleModels: apply
import MLJBase

import MLJModelInterface
const MMI = MLJModelInterface


MMI.@mlj_model mutable struct SequentialCoveringLearner <: MLJModelInterface.Probabilistic
    searchmethod::SearchMethod=BeamSearch()
    max_rulebase_length::Union{Nothing,Integer}=nothing
    suppress_parity_warning::Bool=false
end


function MLJBase.fit(model::SequentialCoveringLearner, verbosity, X, y)
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

function MLJBase.predict(model::SequentialCoveringLearner, fitresult, Xnew)
    yhat = apply(fitresult, PropositionalLogiset(Xnew))
    return yhat
end
#=
TODO cambiare tutti i test adattandoli all' interfaccia MLJ


[Easy test] -- Copiare ed incollare nel terminale:

include("ModalDecisionLists/src/mlj.jl")
seqcovering_model = SequentialCoveringLearner()

include("ModalDecisionLists/test/cn2-accuracy.jl")
learned_list = machine(seqcovering_model, iris_df, CategoricalValue.(y_iris))
fit!(learned_list)
MLJBase.predict(learned_list, iris_df)


=#
