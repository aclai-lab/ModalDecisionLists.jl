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
using ModalDecisionLists: BaseCN2
using CategoricalArrays: CategoricalValue, CategoricalArray

@load DecisionTreeClassifier pkg=DecisionTree
tree_model = MLJDecisionTreeInterface.DecisionTreeClassifier(max_depth=2)

###################################### Utility Functions ###################################
############################################################################################

function splitdataset(
    X::PropositionalLogiset,
    y,
    train_size = 0.7
)
    rng = MersenneTwister(1)
    n_instances = nrow(X)
    ntrain = round(Int, n_instances*train_size)

    permutation = randperm(rng, n_instances)

    train_slice = permutation[1:ntrain]
    X_train = slicedataset(X, train_slice)
    y_train = y[train_slice]

    test_slice = permutation[(ntrain+1):end]
    X_test = slicedataset(X, test_slice)
    y_test = y[test_slice]

    return (X_train, Vector{CLabel}(y_train),
                X_test, Vector{CLabel}(y_test))
end

function preprocess_dataframe(
    X::AbstractDataFrame,
    y
)
    allunique(X) && return (X, y)

    nonunique_ind = nonunique(X)

    return X[findall((!).(nonunique_ind)), :], y[findall((!).(nonunique_ind))]

end
############################################################################################
############################################################################################

# SyntheticDatasets.jl
#=
Xy = RDatasets.dataset("psych", "sat.act")
Xy = RDatasets.dataset("ISLR", "Smarket")
Xy = RDatasets.dataset("ISLR", "Default")
Xy = RDatasets.dataset("Ecdat", "BudgetFood")

=#



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
iris_df = dataset("datasets", "iris")
println("""\n##########################################################\n Iris""")
y_iris = iris_df[:, :Species]
select!(iris_df, Not([:Species]));
X_iris = PropositionalLogiset(iris_df)

dl_iris =  ModalDecisionLists.sole_cn2(X_iris, y_iris)
outcomes = apply(dl_iris, X_iris)
println("\t- full training:\t", MLJ.accuracy(outcomes, y_iris))

X_train, y_train, X_test, y_test = splitdataset(X_iris, y_iris, 0.7)
dl_train_iris = ModalDecisionLists.sole_cn2(X_train, y_train)
outcomes = apply(dl_train_iris, X_test)
println("\t- 2/3 train:\t\t", MLJ.accuracy(outcomes, y_test),"\n")

println("""⊠ truerfirst=true\n""")

dl_iris =  ModalDecisionLists.sole_cn2(X_iris, y_iris; truerfirst=true)
outcomes = apply(dl_iris, X_iris)
println("\t- full training:\t", MLJ.accuracy(outcomes, y_iris))

dl_train_iris = ModalDecisionLists.sole_cn2(X_train, y_train; truerfirst=true)
outcomes = apply(dl_train_iris, X_test)
println("\t- 2/3 train:\t\t", MLJ.accuracy(outcomes, y_test))
println("""\n___________________________________________""")
#
println("""⊠ MLJ - DecisionTree\n""")
learned_tree = machine(tree_model, SoleData.gettable(X_train), CategoricalValue.(y_train))
fit!(learned_tree)
yhat = mode.(MLJ.predict(learned_tree, SoleData.gettable(X_test)))
println("\t- 2/3 train:\t\t", MLJ.accuracy(yhat, y_test))


# 69×5 DataFrame
#  Row │ Column1  dur      gly      bmi      ret
#      │ Int64    Float64  Float64  Float64  Int64
# ─────┼───────────────────────────────────────────
#    1 │       1     10.3     13.7     23.8      0
#    2 │       2      9.9     13.5     23.5      0
#    3 │       3     15.6     13.8     24.8      0
#    4 │       4     26.0     13.0     21.6      1
#      │      ⋮        ⋮        ⋮      ⋮      ⋮
#  666 │     666      6.8     14.6     15.2      0
#  667 │     667      5.8     11.9     32.8      1
#  668 │     668      4.9     15.9     18.8      1
#  669 │     669     10.1     10.1     26.3      0
#                                  648 rows omitted
wesdr = RDatasets.dataset("gamair", "wesdr")
println("""\n##########################################################\n Wesdr""")

y_wesdr = wesdr[:, :ret]
wesdr_df = select(wesdr, Not([:Column1, :ret]));
wesdr_df, y_wesdr = preprocess_dataframe(wesdr_df, y_wesdr)
X_wesdr = PropositionalLogiset(wesdr_df)
dl_wesdr = ModalDecisionLists.sole_cn2(X_wesdr, y_wesdr)

outcomes = apply(dl_wesdr, X_wesdr)
println("\t- full training:\t", MLJ.accuracy(outcomes, y_wesdr))

X_train, y_train, X_test, y_test = splitdataset(X_wesdr, y_wesdr, 0.7)
dl_test_wesdr = ModalDecisionLists.sole_cn2(X_train, y_train)
outcomes = apply(dl_test_wesdr, X_test)
println("\t- 2/3 train:\t\t", MLJ.accuracy(outcomes, y_test),"\n")

println("""⊠ truerfirst=true\n""")

dl_wesdr = ModalDecisionLists.sole_cn2(X_wesdr, y_wesdr; truerfirst=true)
outcomes = apply(dl_wesdr, X_wesdr)
println("\t- full training:\t", MLJ.accuracy(outcomes, y_wesdr))

dl_test_wesdr = ModalDecisionLists.sole_cn2(X_train, y_train; truerfirst=true)
outcomes = apply(dl_test_wesdr, X_test)
println("\t- 2/3 train:\t\t", MLJ.accuracy(outcomes, y_test))
println("""\n___________________________________________""")
#
println("""⊠ MLJ - DecisionTree\n""")
learned_tree = machine(tree_model, SoleData.gettable(X_train), CategoricalArray(y_train))
fit!(learned_tree)
yhat = mode.(MLJ.predict(learned_tree, SoleData.gettable(X_test)))
println("\t- 2/3 train:\t\t", MLJ.accuracy(yhat, y_test))





# wage:
# gross hourly wage rate in euro
# educ:
# education level from 1 [low] to 5 [high]
# exper
# years of experience
# sex
# a factor with levels (males,female)

# 1472×4 DataFrame
#   Row │ Wage      Educ   Exper  Sex
#       │ Float64   Int32  Int32  Cat…
# ──────┼────────────────────────────────
#     1 │  7.78021      1     23  male
#     2 │  4.8185       1     15  female
#     3 │ 10.5636       1     31  male
#     4 │  7.04243      1     32  male
#   ⋮   │    ⋮        ⋮      ⋮      ⋮
#  1470 │ 11.0855       5     19  female
#  1471 │ 15.6498       5     15  female
#  1472 │ 15.0238       5     24  male
#                       1465 rows omitted
bwages = RDatasets.dataset("Ecdat", "Bwages");
println("""\n##########################################################\n Bwages""")
y_bwages = bwages[:, :Sex];
bwages_df = select(bwages, Not([:Sex]));
bwages_df, y_bwages = preprocess_dataframe(bwages_df, y_bwages);
X_bwages = PropositionalLogiset(bwages_df);
dl_bwages = ModalDecisionLists.sole_cn2(X_bwages, y_bwages);

outcomes = apply(dl_bwages, X_bwages);
println("\t- full training:\t", MLJ.accuracy(outcomes, y_bwages),"\n")

X_train, y_train, X_test, y_test = splitdataset(X_bwages, y_bwages, 0.7);
dl_test_bwages = ModalDecisionLists.sole_cn2(X_train, y_train);
outcomes = apply(dl_test_bwages, X_test);
println("\t- 2/3 train:\t\t", MLJ.accuracy(outcomes, y_test))

println("""⊠ truerfirst=true\n""")

y_bwages = bwages[:, :Sex];
bwages_df = select(bwages, Not([:Sex]));
bwages_df, y_bwages = preprocess_dataframe(bwages_df, y_bwages);
X_bwages = PropositionalLogiset(bwages_df);
dl_bwages = ModalDecisionLists.sole_cn2(X_bwages, y_bwages; truerfirst=true);

outcomes = apply(dl_bwages, X_bwages);
println("\t- full training:\t", MLJ.accuracy(outcomes, y_bwages),"\n")

X_train, y_train, X_test, y_test = splitdataset(X_bwages, y_bwages, 0.7);
dl_test_bwages = ModalDecisionLists.sole_cn2(X_train, y_train; truerfirst=true);
outcomes = apply(dl_test_bwages, X_test);
println("\t- 2/3 train:\t\t", MLJ.accuracy(outcomes, y_test))
println("""\n___________________________________________""")
#
println("""⊠ MLJ - DecisionTree\n""")
learned_tree = machine(tree_model, SoleData.gettable(X_train), CategoricalValue.(y_train))
fit!(learned_tree)
yhat = mode.(MLJ.predict(learned_tree, SoleData.gettable(X_test)))
println("\t- 2/3 train:\t\t", MLJ.accuracy(yhat, y_test))
