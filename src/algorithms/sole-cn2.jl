using SoleBase
using SoleBase: CLabel
using SoleLogics
using SoleLogics: nconjuncts, pushconjunct!
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, AbstractAlphabet, BoundedScalarConditions
import SoleData: alphabet
using SoleModels
using SoleModels: DecisionList, Rule, ConstantModel
using SoleModels: default_weights, balanced_weights, bestguess
using DataFrames
using StatsBase: mode, countmap
using FillArrays
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

abstract type SearchMethod end

struct BeamSearch <: SearchMethod end

function maptointeger(y::AbstractVector{<:CLabel})

    values = unique(y)
    integer_y = zeros(Int64, length(y))

    for (i, v) in enumerate(values)
        integer_y[y.==v] .= i
    end
    return integer_y
end

function soleentropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector = default_weights(length(y));
)
    isempty(y) && return Inf
    distribution = values((w isa Ones ? countmap(y) : countmap(y, w)))
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end

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
    sortantecedents(
        antecedents::Vector{Tuple{RuleAntecedent, SatMask}},
        y::AbstractVector{CLabel},
        w::AbstractVector,
        beam_width::Integer,
        quality_evaluator::Function
    )


Sorts rule antecedents based on their quality using a specified evaluation function.

Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *quality evaluator* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the quality
value of the best one

"""
function sortantecedents(
    antecedents::AbstractVector{T},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    quality_evaluator::Function,
)::Tuple{Vector{Int},<:Real} where {
    T<:Tuple{RuleAntecedent, BitVector},
}
    isempty(antecedents) && return [], Inf

    antsquality = map(antd->begin
            satinds = antd[2]
            quality_evaluator(y[satinds], w[satinds])
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
    alph::BoundedScalarConditions,
    antecedent_info::Tuple{RuleAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}

    antecedent, ant_mask = antecedent_info
    conditions = Atom{ScalarCondition}.(atoms(alph))

    filtered_conditions = [(a, check(a, X)) for a ∈ conditions if a ∉ atoms(antecedent)]

    optimizd_conditions = [(a, atom_mask) for (a, atom_mask) ∈ filtered_conditions if
                                            begin
                                                new_antmask = ant_mask .& atom_mask
                                                new_antmask != ant_mask
                                            end]

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
    possible_conditions = optimize ? filteralphabetoptimized(X, alph, antecedent_info) :
                                filteralphabet(X, alph, antecedent)

    return possible_conditions
end


"""
    specializeantecedents(
        antecedents::Vector{Tuple{RuleAntecbedent,SatMask}},
        X::PropositionalLogiset,
        max_rule_length::Union{Nothing,Integer} = nothing,
    )::Vector{Tuple{RuleAntecedent, SatMask}}

Specializes rule antecedents in *antecedents* based on available instances in *X*.

"""
function specializeantecedents(
    antecedents::Vector{Tuple{RuleAntecedent,SatMask}},
    X::PropositionalLogiset,
    max_rule_length::Union{Nothing,Integer}=nothing,
)::Vector{Tuple{RuleAntecedent, SatMask}}

    if isempty(antecedents)

        _alph = alphabet(X)
        _ninst = ninstances(X)

        specialized_ants = Vector{Tuple{RuleAntecedent, SatMask}}([])
        for gfc in SoleData.grouped_featconditions(_alph)
            atomslist = SoleData._atoms(gfc) # TODO rename function _atoms

            (metacondition, _) = gfc

            op = metacondition.test_operator
            (SoleData.isordered(op) && SoleData.polarity(op)) &&
                (atomslist = Iterators.reverse(atomslist))

            uncoveredslice = collect(1:_ninst)
            antecedent_satindexes = zeros(Bool, _ninst)

            partial_antslist = Vector{Tuple{RuleAntecedent, SatMask}}([])
            for _atom in atomslist
                atom_satindexes = check(_atom, slicedataset(X, uncoveredslice; return_view = false))

                antecedent_satindexes[uncoveredslice] = atom_satindexes
                uncoveredslice = uncoveredslice[map(!, atom_satindexes)]

                # @show _atom
                # @show uncoveredslice
                # readline()

                push!(partial_antslist, (RuleAntecedent([_atom]), antecedent_satindexes))
            end

            (SoleData.isordered(op) && SoleData.polarity(op)) &&
                (partial_antslist = Iterators.reverse(partial_antslist))

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
                if !isnothing(max_rule_length) && nconjuncts(antformula) >= max_rule_length
                    continue
                end

                antformula = deepcopy(antformula)

                # new_antcformula = antformula ∧ _atom
                pushconjunct!(antformula, _atom)
                new_antcoverage = antcoverage .& _cov

                push!(specialized_ants, (antformula, new_antcoverage))
            end
        end
        return specialized_ants
    end
end


function findbestantecedent(
    ::BeamSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    beam_width::Integer = 3,
    quality_evaluator::Function = soleentropy,
    max_rule_length::Union{Nothing,Integer} = nothing,
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

    best = (⊤, ones(Int64, nrow(X)))
    best_entropy = quality_evaluator(y, w)

    newcandidates = Tuple{RuleAntecedent, SatMask}[]
    while true
        (candidates, newcandidates) = newcandidates, Tuple{RuleAntecedent, SatMask}[]
        newcandidates = specializeantecedents(candidates, X, max_rule_length)
        # @showlc candidates :green

        isempty(newcandidates) && break

        (perm_, bestcandidate_entropy) = sortantecedents(newcandidates, y, w, beam_width, quality_evaluator)
        newcandidates = newcandidates[perm_]
        if bestcandidate_entropy < best_entropy
            best = newcandidates[1]
            best_entropy = bestcandidate_entropy
        end

        # readline()
        # print("\033c")
    end
    # @show best
    return best
end

function sequentialcovering(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{U},Symbol} = default_weights(length(y));
    search_method::SearchMethod=BeamSearch(),
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    kwargs...
)::DecisionList where {U}

    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]

    w = if isnothing(w) || w == :default
        default_weights(y)
    elseif w == :rebalance
        balanced_weights(y)
    else
        w
    end

    !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
    !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")

    y = y |> maptointeger

    # DEBUG
    # uncoveredslice = collect(1:ninstances(X))

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]
    while true

        bestantecedent, bestantecedent_coverage = findbestantecedent(
            search_method,
            uncoveredX,
            uncoveredy,
            uncoveredw;
            kwargs...
        )
        bestantecedent == ⊤ && break

        rule = begin
            justcoveredy = uncoveredy[bestantecedent_coverage]
            justcoveredw = uncoveredw[bestantecedent_coverage]
            consequent = bestguess(justcoveredy, justcoveredw)
            info_cm = (;
                supporting_labels=collect(justcoveredy),
                supporting_weights=collect(justcoveredw)
            )
            consequent_cm = ConstantModel(consequent, info_cm)
            Rule(bestantecedent, consequent_cm)
        end

        push!(rulebase, rule)

        uncoveredX = slicedataset(uncoveredX, (!).(bestantecedent_coverage); return_view = true)
        uncoveredy = @view uncoveredy[(!).(bestantecedent_coverage)]
        uncoveredw = @view uncoveredw[(!).(bestantecedent_coverage)]

        if !isnothing(max_rulebase_length) && length(rulebase) > max_rulebase_length
            break
        end

        # setdiff!(uncoveredslice, uncoveredslice[bestantecedent_coverage])
        # uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
        # uncoveredy = @view y[uncoveredslice]
        # uncoveredw = @view w[uncoveredslice]
    end
    if !allunique(uncoveredy)
        error("Default class can't be created")
    end
    defaultconsequent = SoleModels.bestguess(uncoveredy)
    return DecisionList(rulebase, defaultconsequent)
end

function sole_cn2(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector,Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; search_method=BeamSearch(), kwargs...)
end

#= Int.(values(currentrule_distribution)) =#
# currentrule_distribution = Dict(unique(y) .=> 0)

# for c in coveredy
#     currentrule_distribution[c] += 1
# end
