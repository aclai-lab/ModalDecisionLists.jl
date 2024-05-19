using Test
using Printf
using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, bestguess
using SoleData
using MLJ
using StatsBase
using Random
using RDatasets
using MLJDecisionTreeInterface
using ModalDecisionLists
using ModalDecisionLists: BaseCN2, MLJInterface
using ModalDecisionLists: preprocess_inputdata
using ModalDecisionLists.Measures: laplace_accuracy
using CategoricalArrays: CategoricalValue, CategoricalArray
using CSV

############################################################################################
const MLJI = MLJInterface
@load DecisionTreeClassifier pkg=DecisionTree

tree_model = MLJDecisionTreeInterface.DecisionTreeClassifier(max_depth=-1)
list_model = MLJI.SequentialCoveringLearner(;beam_width=3)
# TODO add to benchmark
# list_model = MLJI.SequentialCoveringLearner(;beam_width=10)
# list_model = MLJI.SequentialCoveringLearner(;beam_width=20)

_rng = MersenneTwister(16)
_partition = 0.7

############################################################################################

    printstyled("\n iris \n\n", color=:red,bold=true)

    table = dataset("datasets", "iris")
    y = table[:, :Species] |> CategoricalArray
    X = select(table, Not([:Species]));

    X, y = preprocess_inputdata(X,y)

    mach = machine(list_model, X, y);
    # Full training
    fit!(mach; verbosity=1)
    yhat = MLJ.predict(mach, X)
    printstyled("Full training accuracy: ",
                    trunc(MLJ.accuracy(y, yhat),digits=3),"\n",
                    color=:blue,bold=true)

    ########################################################################################
    ##########################  Partial training ###########################################
    ########################################################################################
    train, test = partition(eachindex(y), _partition; rng=_rng)

    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=1,
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=5,
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=15,
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=25,
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)

    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=5,
                                                    truerfirst=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=15,
                                                    truerfirst=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=25,
                                                    truerfirst=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=5,
                                                    discretizedomain=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=15,
                                                    discretizedomain=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=25,
                                                    discretizedomain=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)



#    Partial training on decision tree
   learned_tree = machine(tree_model, X, y)
   fit!(learned_tree, rows=train)
   yhat = mode.(MLJ.predict(learned_tree, X[test, :]))
   printstyled("Partial training (0.7) accuracy: ",
                   trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                   color=:green,bold=true)


############################################################################################

    printstyled("\n biopsy \n\n", color=:red,bold=true)

    table = dataset("MASS", "biopsy")
    y = table[:, :Class] |> CategoricalArray
    X = select(table, Not([:ID, :Class]));

    X, y = preprocess_inputdata(X,y)
    train, test = partition(eachindex(y), _partition; rng=_rng)

    mach = machine(list_model, X, y);
    # Full training
    fit!(mach; verbosity=0)
    yhat = MLJ.predict(mach, X)
    printstyled("Full training accuracy: ",
                    trunc(MLJ.accuracy(y, yhat),digits=3),"\n",
                    color=:blue,bold=true)

    ########################################################################################
    ##########################  Partial training ###########################################
    ########################################################################################
    train, test = partition(eachindex(y), _partition; rng=_rng)


    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=5,
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=15,
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=5,
                                                    truerfirst=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=15,
                                                    truerfirst=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)

    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=5,
                                                    discretizedomain=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)
    ##
    list_model = MLJI.SequentialCoveringLearner(;
                                                    beam_width=15,
                                                    discretizedomain=true
    )
    mach = machine(list_model, X, y);
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
    printstyled("Partial training ($(_partition)) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:blue,bold=true)


    # Partial training on decision tree
    learned_tree = machine(tree_model, X, y)
    fit!(learned_tree, rows=train)
    yhat = mode.(MLJ.predict(learned_tree, X[test, :]))
    printstyled("Partial training (0.7) accuracy: ",
                    trunc(MLJ.accuracy(y[test], yhat), digits=3),"\n",
                    color=:green,bold=true)

############################################################################################
    list_model = MLJI.SequentialCoveringLearner(;
        beam_width=15,
        discretizedomain=true
    )

    printstyled("\n Bwages \n\n", color=:red,bold=true)

    table = dataset("Ecdat", "Bwages")
    y = table[:, :Sex] |> CategoricalArray
    X = select(table, Not([:Sex]));

    X, y = preprocess_inputdata(X,y)

    mach = machine(list_model, X, y);
    # Full training
    # fit!(mach; verbosity=0)
    fit!(mach)
    yhat = MLJ.predict(mach, X)
    printstyled("Full training accuracy: ",
                    trunc(MLJ.accuracy(y, yhat),digits=3),"\n",
                    color=:blue,bold=true)
    # Partial training
    train, test = partition(eachindex(y), _partition; rng=_rng)
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
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


############################################################################################

    printstyled("\n sat.act \n\n", color=:red,bold=true)

    table = dataset("psych", "sat.act")
    y = table[:, :Gender] |> CategoricalArray
    X = select(table, Not([:Gender]));

    X, y = preprocess_inputdata(X,y)

    mach = machine(list_model, X, y);
    # Full training
    # fit!(mach; verbosity=0)
    fit!(mach)
    yhat = MLJ.predict(mach, X)
    printstyled("Full training accuracy: ",
                    trunc(MLJ.accuracy(y, yhat),digits=3),"\n",
                    color=:blue,bold=true)
    # Partial training
    train, test = partition(eachindex(y), _partition; rng=_rng)
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
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

###########################################################################################
    len=1500
    printstyled("\n abalone \n\n", color=:red,bold=true)

    table = CSV.read("datasets/yeast.csv", DataFrame)
    y = table[:, :localization_site] |> CategoricalArray
    X = select(table, Not([:localization_site, :Sequence_Name]));
    X, y = preprocess_inputdata(X[1:len, :],y[1:len])

    mach = machine(list_model, X, y);
    # Full training
    # fit!(mach; verbosity=0)
    fit!(mach, verbosity=0)
    yhat = MLJ.predict(mach, X)
    printstyled("Full training accuracy: ",
                    trunc(MLJ.accuracy(y, yhat),digits=3),"\n",
                    color=:blue,bold=true)
    # Partial training
    train, test = partition(eachindex(y), _partition; rng=_rng)
    fit!(mach, rows=train)
    yhat = MLJ.predict(mach, X[test, :])
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


###########################################################################################

# come faccio a far variare tutti i parametri ?
