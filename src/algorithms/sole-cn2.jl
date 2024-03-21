import SoleBase: CLabel
using SoleLogics
import SoleLogics: children
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, BoundedScalarConditions
import SoleData: alphabet
using SoleModels: DecisionList, Rule, ConstantModel
using DataFrames
using StatsBase: mode, countmap
using ModalDecisionLists


istop(lmlf::LeftmostLinearForm) = children(lmlf) == [⊤]


function soleentropy(
    y::AbstractVector{<:CLabel};
)::Float32

    distribution = values(countmap(y))
    isempty(distribution) &&
        return Inf
    length(distribution) == 1 &&
        return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end

# Check condition equivalence
function checkconditionsequivalence(
    φ1::RuleAntecedent,
    φ2::RuleAntecedent,
)::Bool
    return  length(φ1) == length(φ2) &&
            !any(iszero, map( x-> x ∈ atoms(φ1), atoms(φ2)))
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


# function sortantecedents(
#     star::AbstractVector{Tuple{RuleAntecedent, BitVector}},
#     y::AbstractVector{CLabel},
#     beam_width::Int64,
#     quality_evaluator::F,
# ) where {
#     F<:Function
# }
#     isempty(star) && return [], Inf

#     antsquality = map(antd->begin
#             satinds = interpret(antd, X) |> findall
#             quality_evaluator(y[satinds])
#     end, star)

#     i_newstar = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
#     bestantecedent_quality = antsquality[i_newstar[1]]
#     return (i_newstar, bestantecedent_quality)
# end

function sortantecedents(
    antecedenslist::AbstractVector{Tuple{RuleAntecedent, BitVector}},
    y::AbstractVector{CLabel},
    beam_width::Int64,
    quality_evaluator::F,
) where {
    F<:Function
}
    isempty(antecedenslist) && return [], Inf

    antsquality = map(antd->begin
            satinds = antd[2]
            quality_evaluator(y[satinds])
    end, antecedenslist)

    i_newstar = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
    bestantecedent_quality = antsquality[i_newstar[1]]
    return (i_newstar, bestantecedent_quality)
end



function newconditions(
    X::PropositionalLogiset,
    antecedent_param::Tuple{RuleAntecedent, BitVector}
)

    (antecedent, satindexes) = antecedent_param

    coveredX = slicedataset(X, satindexes; return_view = true)
    # @show coveredX


    conditions = Atom{ScalarCondition}.(atoms(alphabet(coveredX)))
    ### la copertura dei nuovi atomi la calcolo su X e NON su coveredX ###
    possible_conditions = [(a, interpret(a, X)) for a in conditions if a ∉ atoms(antecedent)]

    return possible_conditions
end


pushchildren!(φ::RuleAntecedent, a::Atom) = push!(φ.children, a)

function specializeantecedents(
    star::AbstractVector{Tuple{RuleAntecedent, BitVector}},
    X::PropositionalLogiset,
)::Vector{Tuple{RuleAntecedent, BitVector}}

    if isempty(star)
        conditions =  map(sc->Atom{ScalarCondition}(sc), alphabet(X))
        return  map(a->(RuleAntecedent([a]), interpret(a, X)), conditions)
    else
        newstar = Vector{Tuple{RuleAntecedent, BitVector}}([])
        for i_ant ∈ star

            # Vector{Tuple{Atom, BitVector}}
            i_possibleconditions = newconditions(X, i_ant)

            # @showlc atoms(i_ant[1]) :red
            # @showlc i_possibleconditions :blue

            isempty(i_possibleconditions) && continue

            for (j_atom, j_cov) ∈ i_possibleconditions

                (i_formula, i_coverage) = deepcopy(i_ant)

                pushchildren!(i_formula, j_atom)
                newcoverage = i_coverage .& j_cov

                push!(newstar, (i_formula, newcoverage))
            end
        end
    end
    return newstar
end

function beamsearch(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    beam_width::Integer,
    quality_evaluator::F
)::LeftmostConjunctiveForm where{
    F<:Function
}

    bestantecedent = (LeftmostConjunctiveForm([⊤]), ones(Int64, nrow(X)))
    bestantecedent_entropy = quality_evaluator(y)

    newstar = Tuple{RuleAntecedent, BitVector}[]
    while true
        (star, newstar) = newstar, Tuple{RuleAntecedent, BitVector}[]
        newstar = specializeantecedents(star, X)
        # @showlc star :green

        isempty(newstar) && break

        (perm_, candidateantecedent_entropy) = sortantecedents(newstar, y, beam_width, quality_evaluator)
        newstar = newstar[perm_]
        if candidateantecedent_entropy < bestantecedent_entropy
            bestantecedent = newstar[1]
            bestantecedent_entropy = candidateantecedent_entropy
        end

        # readline()
        # print("\033c")
    end
    # @show bestantecedent
    return bestantecedent[begin]
end

function sequentialcovering(
    X::PropositionalLogiset,
    y::AbstractVector{CLabel};
    beam_width::Integer = 3,
    quality_evaluator::Function = soleentropy,
)::DecisionList
    length(y) != nrow(X) && error("size of X and y mismatch")
    uncoveredslice = collect(1:ninstances(X))

    uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
    uncoveredy = y[uncoveredslice]

    rulelist = Rule[]
    while true

        bestantecedent = beamsearch(uncoveredX, uncoveredy, beam_width, quality_evaluator)
        istop(bestantecedent) && break

        antecedentcoverage  = interpret(bestantecedent, uncoveredX) |> findall
        consequent = uncoveredy[antecedentcoverage] |> mode
        info_cm = (;
            supporting_labels = collect(uncoveredy)
        )
        consequent_cm = ConstantModel(consequent, info_cm)

        push!(rulelist, Rule(bestantecedent, consequent_cm))

        setdiff!(uncoveredslice, uncoveredslice[antecedentcoverage])
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
