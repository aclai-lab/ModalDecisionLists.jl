using Test
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

# @load DecisionTreeClassifier pkg=DecisionTree
# tree_model = MLJDecisionTreeInterface.DecisionTreeClassifier(max_depth=-1)

const MLJI = MLJInterface
list_model = MLJI.SequentialCoveringLearner()


# 150×5 DataFrame
#  Row │ SepalLength  SepalWidth  PetalLength  PetalWidth  Species
#      │ Float64      Float64     Float64      Float64     Cat…
# ─────┼─────────────────────────────────────────────────────────────
#    1 │         5.1         3.5          1.4         0.2  setosa
#    2 │         4.9         3.0          1.4         0.2  setosa
#    3 │         4.7         3.2          1.3         0.2  setosa
#    4 │         4.6         3.1          1.5         0.2  setosa
#      │      ⋮           ⋮            ⋮           ⋮           ⋮
#  148 │         6.5         3.0          5.2         2.0  virginica
#  149 │         6.2         3.4          5.4         2.3  virginica
#  150 │         5.9         3.0          5.1         1.8  virginica
#                                                    143 rows omitted
iris = dataset("datasets", "iris")
# println("""\n##########################################################\n Iris""")
y_iris = iris[:, :Species]
X_iris = select(iris, Not([:Species]));
learned_list = machine(list_model, X_iris, CategoricalValue.(y_iris))
fit!(learned_list)
yhat = MLJ.predict(learned_list, X_iris)
# #
# println("""⊠ MLJ - DecisionTree\n""")
# learned_tree = machine(tree_model, SoleData.gettable(X_train), CategoricalValue.(y_train))
# fit!(learned_tree)
# yhat = mode.(MLJ.predict(learned_tree, SoleData.gettable(X_test)))
# println("\t- 2/3 train:\t\t", MLJ.accuracy(yhat, y_test))
