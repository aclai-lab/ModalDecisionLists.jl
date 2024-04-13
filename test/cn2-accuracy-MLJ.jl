using Test
using Printf
using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, orange_decision_list
using SoleData
using MLJ
using StatsBase
using Random
using RDatasets
using ModalDecisionLists
using ModalDecisionLists: BaseCN2, MLJInterface
using CategoricalArrays: CategoricalValue, CategoricalArray

"""
Dumb utility function to preprocess input data:
    * remove duplicated rows
    * remove rows with missing values
"""
function preprocess_inputdata(
    X::AbstractDataFrame,
    y
)
    allunique(X) && return (X, y)
    nonunique_ind = nonunique(X)
    Xy = hcat( X[findall((!).(nonunique_ind)), :],
               y[findall((!).(nonunique_ind))]
    ) |> dropmissing
    return Xy[:, 1:(end-1)], Xy[:, end]
end

############################################################################################
const MLJI = MLJInterface
@load DecisionTreeClassifier pkg=DecisionTree

tree_model = MLJDecisionTreeInterface.DecisionTreeClassifier(max_depth=-1)
list_model = MLJI.SequentialCoveringLearner()

_rng = MersenneTwister(16)
_partition = 0.7

Xy = RDatasets.dataset("psych", "sat.act")

table_ntuples = [
    (package = "datasets", tablename = "iris",
        target  = :Species,
        exclude = [:Species]),
    (package = "gamair", tablename = "wesdr",
        target  = :ret,
        exclude = [:Column1, :ret]),
    (package = "Ecdat", tablename = "Bwages",
        target  = :Sex,
        exclude = [:Sex]),
    (package = "psych", tablename = "sat.act",
        target  = :Gender,
        exclude = [:Gender]),
    (package = "ISLR", tablename = "Smarket",
        target  = :Direction,
        exclude = [:Direction]),
]

############################################################################################


for table_nt in table_ntuples

    printstyled("\n $(table_nt.tablename) \n\n", color=:red,bold=true)

    table = dataset(table_nt.package, table_nt.tablename)
    y = table[:, table_nt.target] |> CategoricalArray
    X = select(table, Not([table_nt.target]));

    X, y = preprocess_inputdata(X,y)

    learned_machine = machine(list_model, X, y);
    # Full training
    fit!(learned_machine)
    yhat = MLJ.predict(learned_machine, X)
    printstyled("Full training accuracy: ",
                    trunc(MLJ.accuracy(y, yhat),digits=3),"\n",
                    color=:blue,bold=true)
    # Partial training
    train, test = partition(eachindex(y), _partition; rng=_rng)
    fit!(learned_machine, rows=train)
    yhat = MLJ.predict(learned_machine, X[test, :])
    printstyled("Partial training (0.7) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)

    # Partial training on decision tree
    learned_tree = machine(tree_model, X, y)
    fit!(learned_tree, rows=train)
    yhat = mode.(MLJ.predict(learned_tree, X[test, :]))
    printstyled("Partial training (0.7) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:green,bold=true)
end
