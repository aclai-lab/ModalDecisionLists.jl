import SoleBase: CLabel
using SoleLogics
import SoleLogics: children
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, AbstractAlphabet, BoundedScalarConditions
import SoleData: alphabet
using SoleModels: DecisionList, Rule, ConstantModel
using DataFrames
using StatsBase: mode, countmap
using ModalDecisionLists
using MLJ
using Debugger

const RuleAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom{ScalarCondition}}
const SatMask = BitVector

X...,y = MLJ.load_iris()
X_df = DataFrame(X)
X = PropositionalLogiset(X_df)
n_instances = ninstances(X)
y = Vector{CLabel}(y)

_alphabet = alphabet(X)
gfc_1 = _alphabet.grouped_featconditions[1]

atoms_1 = Vector{Atom{ScalarCondition}}(atoms(gfc_1))

function efficentcoverage(
    atomslist::Vector{Atom{ScalarCondition}},
    Xpl::PropositionalLogiset
)::Vector{Tuple{RuleAntecedent, SatMask}}

    X_nrows = ninstances(Xpl)
    antecedents = Vector{Tuple{RuleAntecedent, SatMask}}([])

    uncoveredslice = collect(1:X_nrows)
    antecedent_cov = zeros(Bool, X_nrows)
    for _atom in atomslist

        atom_cov = check(_atom, slicedataset(Xpl, uncoveredslice; return_view = false))

        antecedent_cov[uncoveredslice] = atom_cov
        uncoveredslice = uncoveredslice[map(!, atom_cov)]
        push!(antecedents, (RuleAntecedent([_atom]), antecedent_cov))
    end
    return antecedents
end


antecedents_old =  map(a->(RuleAntecedent([a]), check(a, X)), atoms_1)
antecedents_new = efficentcoverage(atoms_1, X)

for (i_new, i_old) in zip(antecedents_new, antecedents_old)
    println(findall(i_new[2]) == findall(i_old[2]))
end

using BenchmarkTools
@benchmark map(a->(RuleAntecedent([a]), check(a, X)), atoms_1)
@benchmark efficentcoverage(atoms_1, X)
