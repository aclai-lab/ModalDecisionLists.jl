import SoleBase: CLabel
using SoleLogics
import SoleLogics: children
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, AbstractAlphabet, BoundedScalarConditions
import SoleData: alphabet
using SoleModels: DecisionList, Rule, ConstantModel
using DataFrames
using StatsBase: mode, countmap, counts
using ModalDecisionLists

#=


using Test
using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, orange_decision_list
using SoleData
using MLJ
using StatsBase
using Random
import SoleData: PropositionalLogiset

module BaseCN2
include("../src/algorithms/base-cn2.jl")
end

module SoleCN2
include("../src/algorithms/sole-cn2.jl")
end

# Input
X...,y = MLJ.load_iris()
X_df = DataFrame(X)
X = PropositionalLogiset(X_df)
n_instances = ninstances(X)
y = Vector{CLabel}(y)

=#




istop(lmlf::LeftmostLinearForm) = children(lmlf) == [⊤]

pushchildren!(φ::RuleAntecedent, a::Atom) = push!(φ.children, a) |> RuleAntecedent

function maptointeger(y::AbstractVector{<:CLabel})

    values = unique(y)
    integer_y = zeros(Int64, length(y))

    for (i,v) in enumerate(values)
        integer_y[y .== v] .= i
    end
    return integer_y
end

function soleentropy(y::AbstractVector{Int64};)::Float32

    isempty(y) && return Inf

    distribution = counts(y)

    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end

# Check condition equivalence


# function feature(
#     φ::RuleAntecedent
# )::AbstractVector{UnivariateSymbolValue}
#     conditions = value.(atoms(φ))
#     return feature.(conditions)
# end
# function conjunctibleconds(
#     bsc::BoundedScalarConditions,
#     φ::Formula
# )::Union{Bool,BoundedScalarConditions}

#     φ_features = feature(φ)
#     conds = [(meta_cond, vals) for (meta_cond, vals) ∈ bsc.grouped_featconditions
#                 if feature(meta_cond) ∉ φ_features]
#     return conds == [] ? false : BoundedScalarConditions{ScalarCondition}(conds)
# end

"""
    function sortantecedents(
        antecedents::Vector{Tuple{RuleAntecedent, SatMask}},
        y::AbstractVector{CLabel},
        beam_width::Integer,
        quality_evaluator<:Function,


Sorts rule antecedents based on their quality using a specified evaluation function.

Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *quality evaluator* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the quality
value of the best one

"""
function sortantecedents(
    antecedents::AbstractVector{Tuple{RuleAntecedent, SatMask}},
    y::AbstractVector{<:CLabel},
    beam_width::Integer,
    quality_evaluator::F,
)::Tuple{Vector{Int},<:Real} where {
    T<:Tuple{RuleAntecedent, BitVector},
    F<:Function
}
    isempty(antecedents) && return [], Inf

    antsquality = map(antd->begin
            satinds = antd[2]
            quality_evaluator(y[satinds])
    end, antecedents)

    newstar_perm = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
    bestantecedent_quality = antsquality[newstar_perm[1]]

    return (newstar_perm, bestantecedent_quality)
end



"""
    newatoms(
        X::PropositionalLogiset,
        antecedent::Tuple{RuleAntecedent, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns a list of all possible conditions (atoms) that can be generated from instances of X
and can further specialize the input formula. Each condition is decorated by a bitmask
indicating which examples in X satisfy that condition."

"""

function filteralphabet(
    X::PropositionalLogiset,
    alph::AbstractAlphabet,
    antecedent::RuleAntecedent
)::Vector{Tuple{Atom, SatMask}}

    conditions = Atom{ScalarCondition}.(atoms(alph))
    possible_conditions = [(a, check(a, X)) for a in conditions if a ∉ atoms(antecedent)]

    return possible_conditions
end

function filteralphabetoptimized(
    X::PropositionalLogiset,
    alphabet::AbstractAlphabet,
    antecedent::Tuple{RuleAntecedent, SatMask}
)::Vector{Tuple{Atom, SatMask}}
    return [(a, check(a, X)) for a in alphabet if a ∉ atoms(antecedent)]
end

function filteralphabetoptimized(
    X::PropositionalLogiset,
    alphabet::BoundedScalarConditions,
    antecedent::Tuple{RuleAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}
    return [(a, check(a, X)) for a in alphabet if a ∉ atoms(antecedent)]
end


function newatoms(
    X::PropositionalLogiset,
    antecedent_info::Tuple{RuleAntecedent, BitVector};
    optimize = false
)::Vector{Tuple{Atom{ScalarCondition}, BitVector}}

    (antecedent, satindexes) = antecedent_info
    coveredX = slicedataset(X, satindexes; return_view = true)

    alph = alphabet(coveredX)

    ### la copertura dei nuovi atomi la calcolo su X e NON su coveredX ###
    possible_conditions = optimize ? filteralphabetoptimized(X, alph, antecedent) : filteralphabet(X, alph, antecedent)
    return possible_conditions
end


"""
    function specializeantecedents(
        antecedents::Vector{Tuple{RuleAntecbedent,SatMask}},
        X::PropositionalLogiset,
    )::Vector{Tuple{RuleAntecedent, SatMask}}

Specializes rule antecedents in *antecedents* based on available instances in *X*.

"""
function specializeantecedents(
    antecedents::Vector{Tuple{RuleAntecedent,SatMask}},
    X::PropositionalLogiset,
)::Vector{Tuple{RuleAntecedent, SatMask}}

    if isempty(antecedents)

        _alph = alphabet(X)
        _nrow = ninstances(X)

        specialized_ants = Vector{Tuple{RuleAntecedent, SatMask}}([])
        for gfc in SoleData.featconditions(_alph)

            atomslist = atoms(gfc)

            (metacondition, _) = gfc
            (metacondition.test_operator == (>=)) &&
                reverse!(atomslist)

            uncoveredslice = collect(1:_nrow)
            antecedent_satindexes = zeros(Bool, _nrow)

            partial_antslist = Vector{Tuple{RuleAntecedent, SatMask}}([])
            for _atom in atomslist
                atom_satindexes = check(_atom, slicedataset(X, uncoveredslice; return_view = false))

                antecedent_satindexes[uncoveredslice] = atom_satindexes
                uncoveredslice = uncoveredslice[map(!, atom_satindexes)]
                push!(partial_antslist, (RuleAntecedent([_atom]), antecedent_satindexes))
            end

            metacondition.test_operator == (>=) &&
                reverse!(partial_antslist)

            append!(specialized_ants, partial_antslist)
        end
        return specialized_ants

        return map(a->(RuleAntecedent([a]), check(a, X)), alphabet(X))
    else
        specialized_ants = Tuple{RuleAntecedent, SatMask}[]
        for _ant ∈ antecedents

            # i_conjunctibleatoms refer to all the conditions (Atoms) that can be
            # joined to the i-th antecedent. These are calculated only for the values ​​
            # of the instances already covered by the antecedent.
            conjunctibleatoms = newatoms(X, _ant)

            # @showlc atoms(_ant[1]) :red
            # @showlc i_conjunctibleatoms :blue

            isempty(conjunctibleatoms) && continue

            for (_atom, _cov) ∈ conjunctibleatoms

                (antformula, antcoverage) = _ant
                antformula = deepcopy(antformula)

                # new_antcformula = antformula ∧ _atom
                new_antcformula = pushchildren!(antformula, _atom)
                new_antcoverage = antcoverage .& _cov

                push!(specialized_ants, (new_antcformula, new_antcoverage))
            end
        end
        return specialized_ants
    end
end

function beamsearch(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    beam_width::Integer;
    quality_evaluator::F,
    max_rule_length
)::Tuple{LeftmostConjunctiveForm,SatMask} where {
    F<:Function
}

    bestantecedent = (LeftmostConjunctiveForm([⊤]), ones(Int64, nrow(X)))
    bestantecedent_entropy = quality_evaluator(y)

    newcandidates = Tuple{RuleAntecedent, SatMask}[]
    while true
        (candidates, newcandidates) = newcandidates, Tuple{RuleAntecedent, SatMask}[]
        newcandidates = specializeantecedents(candidates, X)
        # @showlc candidates :green

        isempty(newcandidates) && break

        (perm_, bestcandidate_entropy) = sortantecedents(newcandidates, y, beam_width, quality_evaluator)
        newcandidates = newcandidates[perm_]
        if bestcandidate_entropy < bestantecedent_entropy
            bestantecedent = newcandidates[1]
            bestantecedent_entropy = bestcandidate_entropy
        end

        # readline()
        # print("\033c")
    end
    # @show bestantecedent
    return bestantecedent
end

function sequentialcovering(
    X::PropositionalLogiset,
    y::AbstractVector{CLabel};
    beam_width::Integer = 3,
    quality_evaluator::Function = soleentropy,
    max_rule_length::Union{Nothing,Integer} = nothing,
    max_rulebase_length::Union{Nothing,Integer}=nothing,
)::DecisionList

    length(y) != nrow(X) && error("size of X and y mismatch")
    y = y |> maptointeger

    uncoveredslice = collect(1:ninstances(X))

    uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
    uncoveredy = y

    rulebase = Rule[]
    while true

        bestantecedent, bestantecedent_coverage = beamsearch(
            uncoveredX,
            uncoveredy,
            beam_width;
            quality_evaluator=quality_evaluator,
            max_rule_length=max_rule_length
        )
        istop(bestantecedent) && break

        consequent = uncoveredy[bestantecedent_coverage] |> mode
        info_cm = (;
            supporting_labels = collect(uncoveredy)
        )
        consequent_cm = ConstantModel(consequent, info_cm)

        push!(rulebase, Rule(bestantecedent, consequent_cm))

        if !isnothing(max_rulebase_length) && length(rulebase) > max_rulebase_length
            break
        end

        setdiff!(uncoveredslice, uncoveredslice[bestantecedent_coverage])
        uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
        uncoveredy = @view y[uncoveredslice]
    end
    if !allunique(uncoveredy)
        error("Default class can't be created")
    end
    defaultconsequent = SoleModels.bestguess(uncoveredy)
    return DecisionList(rulebase, defaultconsequent)
end

sole_cn2 = sequentialcovering

#= Int.(values(currentrule_distribution)) =#
# currentrule_distribution = Dict(unique(y) .=> 0)

# for c in coveredy
#     currentrule_distribution[c] += 1
# end
