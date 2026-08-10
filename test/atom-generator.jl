using Test
using DataFrames
using SoleData
using SoleLogics
using ModalDecisionLists
using ModalDecisionLists: AtomGenerator
using MLJ


X, y = @load_iris
X = PropositionalLogiset(DataFrame(X))

alph = alphabet(X; y=ones(Int, ninstances(X)), keep_unique=true, test_operators=[<, >=])
conditions = ModalDecisionLists.alphabet2conditions(AtomGenerator(), UnionAlphabet([alph]), X)

thresholds = [SoleData.threshold(SoleLogics.value(atom)) for (atom, _) in conditions if SoleLogics.value(atom) isa SoleData.ScalarCondition]

conds = [condition[1] for condition in conditions]

println("alphabet: \n\t$alph")
println("\nconditions: \n\t$conds")