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
using ModalDecisionLists: BaseCN2, MLJInterface, sole_laplace_estimator
using CategoricalArrays: CategoricalValue, CategoricalArray

using BenchmarkTools

const MLJI = MLJInterface
list_model = MLJI.SequentialCoveringLearner()

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


# Time
for table_nt in table_ntuples

    printstyled("\n $(table_nt.tablename) \n\n", color=:red,bold=true)

    table = dataset(table_nt.package, table_nt.tablename)
    y = table[:, table_nt.target] |> CategoricalArray
    X = select(table, Not([table_nt.target]));

    learned_machine = machine(list_model, X, y);
    # Full training
    # display(@benchmark fit!(learned_machine) setup=(learned_machine=$learned_machine));
    fit!(learned_machine)
end
