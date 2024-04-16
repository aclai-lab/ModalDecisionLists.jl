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

table_ntuples = [
    (package = "datasets", tablename = "iris",
        target  = :Species,
        exclude = [:Species]),
    #         150×5 DataFrame
    #  Row │ SepalLength  SepalWidth  PetalLength  PetalWidth  Species
    #      │ Float64      Float64     Float64      Float64     Cat…
    # ─────┼─────────────────────────────────────────────────────────────
    #    1 │         5.1         3.5          1.4         0.2  setosa
    #    2 │         4.9         3.0          1.4         0.2  setosa
    #    3 │         4.7         3.2          1.3         0.2  setosa
    #    4 │         4.6         3.1          1.5         0.2  setosa
    #    5 │         5.0         3.6          1.4         0.2  setosa
    #      │      ⋮           ⋮            ⋮           ⋮           ⋮
    #  145 │         6.7         3.3          5.7         2.5  virginica
    #  146 │         6.7         3.0          5.2         2.3  virginica
    #  147 │         6.3         2.5          5.0         1.9  virginica
    #  148 │         6.5         3.0          5.2         2.0  virginica
    #  149 │         6.2         3.4          5.4         2.3  virginica
    #  150 │         5.9         3.0          5.1         1.8  virginica
    (package = "gamair", tablename = "wesdr",
        target  = :ret,
        exclude = [:Column1, :ret]),
    #     669×5 DataFrame
    #     Row │ Column1  dur      gly      bmi      ret
    #         │ Int64    Float64  Float64  Float64  Int64
    #    ─────┼───────────────────────────────────────────
    #       1 │       1     10.3     13.7     23.8      0
    #       2 │       2      9.9     13.5     23.5      0
    #       3 │       3     15.6     13.8     24.8      0
    #       4 │       4     26.0     13.0     21.6      1
    #       5 │       5     13.8     11.1     24.6      1
    #         │    ⋮        ⋮        ⋮        ⋮       ⋮
    #     666 │     666      6.8     14.6     15.2      0
    #     667 │     667      5.8     11.9     32.8      1
    #     668 │     668      4.9     15.9     18.8      1
    #     669 │     669     10.1     10.1     26.3      0

    (package = "Ecdat", tablename = "Bwages",
        target  = :Sex,
        exclude = [:Sex]),
    #         1472×4 DataFrame
    #   Row │ Wage      Educ   Exper  Sex
    #       │ Float64   Int32  Int32  Cat…
    # ──────┼────────────────────────────────
    #     1 │  7.78021      1     23  male
    #     2 │  4.8185       1     15  female
    #     3 │ 10.5636       1     31  male
    #     4 │  7.04243      1     32  male
    #     5 │  7.88752      1      9  male
    #     6 │  8.20006      1     15  female
    #       │    ⋮        ⋮      ⋮      ⋮
    #  1468 │ 11.1196       5      1  male
    #  1469 │  5.45337      5     14  female
    #  1470 │ 11.0855       5     19  female
    #  1471 │ 15.6498       5     15  female
    #  1472 │ 15.0238       5     24  male

    (package = "psych", tablename = "sat.act",
        target  = :Gender,
        exclude = [:Gender]),
    #         700×6 DataFrame
    #  Row │ Gender  Education  Age    ACT    SATV   SATQ
    #      │ Int64   Int64      Int64  Int64  Int64  Int64?
    # ─────┼─────────────────────────────────────────────────
    #    1 │      2          3     19     24    500      500
    #    2 │      2          3     23     35    600      500
    #    3 │      2          3     20     21    480      470
    #    4 │      1          4     27     26    550      520
    #    5 │      1          2     33     31    600      550
    #      │   ⋮         ⋮        ⋮      ⋮      ⋮       ⋮
    #  698 │      2          3     24     31    700      630
    #  699 │      1          4     35     32    700      780
    #  700 │      1          5     25     25    600      600
    (package = "ISLR", tablename = "Smarket",
        target  = :Direction,
        exclude = [:Direction]),
    #     1250×9 DataFrame
    #     Row │ Year     Lag1     Lag2     Lag3     Lag4     Lag5     Volume   Today    Direction
    #         │ Float64  Float64  Float64  Float64  Float64  Float64  Float64  Float64  Cat…
    #   ──────┼───────────────────────────────────────────────────────────────────────────────────
    #       1 │  2001.0    0.381   -0.192   -2.624   -1.055    5.01   1.1913     0.959  Up
    #       2 │  2001.0    0.959    0.381   -0.192   -2.624   -1.055  1.2965     1.032  Up
    #       3 │  2001.0    1.032    0.959    0.381   -0.192   -2.624  1.4112    -0.623  Down
    #       4 │  2001.0   -0.623    1.032    0.959    0.381   -0.192  1.276      0.614  Up
    #       5 │  2001.0    0.614   -0.623    1.032    0.959    0.381  1.2057     0.213  Up
    #         │    ⋮        ⋮        ⋮        ⋮        ⋮        ⋮        ⋮        ⋮         ⋮
    #    1248 │  2005.0   -0.955    0.043    0.422    0.252   -0.024  1.54047    0.13   Up
    #    1249 │  2005.0    0.13    -0.955    0.043    0.422    0.252  1.42236   -0.298  Down
    #    1250 │  2005.0   -0.298    0.13    -0.955    0.043    0.422  1.38254   -0.489  Down
    #                                                                            1197 rows omitted
]

############################################################################################

# come faccio a far variare tutti i parametri ?
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
