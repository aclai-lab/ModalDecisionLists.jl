import SoleBase: CLabel
using SoleLogics
import SoleLogics: children
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, BoundedScalarConditions
import SoleData: alphabet
using SoleModels: DecisionList, Rule, ConstantModel
using DataFrames
using StatsBase: mode, countmap, counts
using ModalDecisionLists


istop(lmlf::LeftmostLinearForm) = children(lmlf) == [⊤]

pushchildren!(φ::RuleAntecedent, a::Atom) = push!(φ.children, a)

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
        antecedenslist::Vector{Tuple{RuleAntecedent, BitVector}},
        y::AbstractVector{CLabel},
        beam_width::Integer,
        quality_evaluator<:Function,


Sorts rule antecedents based on their quality using a specified evaluation function.

Takes an *antecedentslist*, each decorated by a BitVector indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *quality evaluator* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the quality
value of the best one

"""
function sortantecedents(
    antecedenslist::Vector{T},
    y::AbstractVector{Int64},
    beam_width::Integer,
    quality_evaluator::F,
)::Tuple{Vector{Int},<:Real} where {
    T<:Tuple{RuleAntecedent, BitVector},
    F<:Function
}


    isempty(antecedenslist) && return [], Inf

    antsquality = map(antd->begin
            satinds = antd[2]
            quality_evaluator(y[satinds])
    end, antecedenslist)

    newstar_perm = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
    bestantecedent_quality = antsquality[newstar_perm[1]]

    return (newstar_perm, bestantecedent_quality)
end

"""
    newconditions(
        X::PropositionalLogiset,
        antecedent_info::Tuple{RuleAntecedent, BitVector}
    )::Vector{Tuple{Atom{ScalarCondition}, BitVector}}

Returns a list of all possible conditions (atoms) that can be generated from instances of X
and can further specialize the input formula. Each condition is decorated by a bitmask
indicating which examples in X satisfy that condition."

"""
function newconditions(
    X::PropositionalLogiset,
    antecedent_info::Tuple{RuleAntecedent, BitVector}
)::Vector{Tuple{Atom{ScalarCondition}, BitVector}}

    (antecedent, satindexes) = antecedent_info

    coveredX = slicedataset(X, satindexes; return_view = true)
    # @show coveredX
    conditions = Atom{ScalarCondition}.(atoms(alphabet(coveredX)))
    ### la copertura dei nuovi atomi la calcolo su X e NON su coveredX ###
    possible_conditions = [(a, interpret(a, X)) for a in conditions if a ∉ atoms(antecedent)]

    return possible_conditions
end




"""
    function specializeantecedents(
        ants_tospecialize::Vector{Tuple{RuleAntecedent,BitVector}},
        X::PropositionalLogiset,
    )::Vector{Tuple{RuleAntecedent, BitVector}}

Specializes rule antecedents in *ants_tospecialize* based on available instances in *X*.

"""
function specializeantecedents(
    ants_tospecialize::Vector{T},
    X::PropositionalLogiset,
)::Vector{T} where{
    T<:Tuple{RuleAntecedent,BitVector}
}

    if isempty(ants_tospecialize)
        conditions =  map(sc->Atom{ScalarCondition}(sc), alphabet(X))
        return  map(a->(RuleAntecedent([a]), interpret(a, X)), conditions)
    else
        specialized_ants = Tuple{RuleAntecedent, BitVector}[]
        for _ant ∈ ants_tospecialize

            possibleconditions = newconditions(X, _ant)

            # @showlc atoms(i_ant[1]) :red
            # @showlc i_possibleconditions :blue

            isempty(possibleconditions) && continue

            for (_atom, _cov) ∈ possibleconditions

                (antformula, antcoverage) = deepcopy(_ant)

                new_antcformula = antformula ∧ _atom |> LeftmostConjunctiveForm
                new_antcoverage = antcoverage .& _cov

                push!(specialized_ants, (new_antcformula, new_antcoverage))
            end
        end
    end
    return specialized_ants
end

function beamsearch(
    X::PropositionalLogiset,
    y::AbstractVector{Int64},
    beam_width::Integer,
    quality_evaluator::F
)::Tuple{LeftmostConjunctiveForm,BitVector} where {
    F<:Function
}

    bestantecedent = (LeftmostConjunctiveForm([⊤]), ones(Int64, nrow(X)))
    bestantecedent_entropy = quality_evaluator(y)

    newcandidates = Tuple{RuleAntecedent, BitVector}[]
    while true
        (candidates, newcandidates) = newcandidates, Tuple{RuleAntecedent, BitVector}[]
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
)::DecisionList

    length(y) != nrow(X) && error("size of X and y mismatch")
    y = y |> maptointeger

    uncoveredslice = collect(1:ninstances(X))

    uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
    uncoveredy = y

    rulelist = Rule[]
    while true

        bestantecedent, bestantecedent_coverage = beamsearch(uncoveredX, uncoveredy, beam_width, quality_evaluator)
        istop(bestantecedent) && break

        consequent = uncoveredy[bestantecedent_coverage] |> mode
        info_cm = (;
            supporting_labels = collect(uncoveredy)
        )
        consequent_cm = ConstantModel(consequent, info_cm)

        push!(rulelist, Rule(bestantecedent, consequent_cm))

        setdiff!(uncoveredslice, uncoveredslice[bestantecedent_coverage])
        uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
        uncoveredy = @view y[uncoveredslice]
    end
    if !allunique(uncoveredy)
        error("Default class can't be created")
    end
    defaultconsequent = uncoveredy[begin]
    return DecisionList(rulelist, defaultconsequent)
end

sole_cn2 = sequentialcovering

#= Int.(values(currentrule_distribution)) =#
# currentrule_distribution = Dict(unique(y) .=> 0)

# for c in coveredy
#     currentrule_distribution[c] += 1
# end
