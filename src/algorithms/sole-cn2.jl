using SoleBase
using SoleBase: CLabel
using SoleLogics
using SoleLogics: nconjuncts, pushconjunct!
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, AbstractAlphabet, UnionAlphabet
import SoleData: alphabet, test_operator, isordered, polarity, turnatoms, grouped_featconditions
using SoleModels
using SoleModels: DecisionList, Rule, ConstantModel
using SoleModels: default_weights, balanced_weights, bestguess
using DataFrames
using StatsBase: mode, countmap, counts, Weights
using FillArrays
using ModalDecisionLists


function maptointeger(y::AbstractVector{<:CLabel})

    # ordered values
    values = unique(y)
    integer_y = zeros(Int64, length(y))

    for (i, v) in enumerate(values)
        integer_y[y.==v] .= i
    end
    return integer_y, values
end

function soleentropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end

############################################################################################
############################################################################################
############################################################################################

"""
Abstract type for all search methods to be used in [`sequentialcovering`](@ref).

findbestantecedent

See also [`BeamSearch`](@ref, [`RandSearch`](@ref).
"""
abstract type SearchMethod end

"""
Per ogni nuovo SearchMethod, findbestantecedent deve essere implementato.
"""
function findbestantecedent(
    ::SearchMethod,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
)
    return error("Please, provide method...")
end

############################################################################################
############## Random search ###############################################################
############################################################################################

"""
Generate random formulas (`SoleLogics.randformula`)
....
"""
@with_kw struct RandSearch <: SearchMethod
    cardinality::Integer=10
    quality_evaluator::Function=soleentropy
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
end

function findbestantecedent(
    sm::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector
)::Tuple{Formula,SatMask}
    bestantecedent = begin
        if !allunique(y)
            randformulas = [
                begin
                    rfa = randformula(sm.syntaxheight, alphabet(X), sm.operators)
                    (rfa, check(rfa, X))
                end for _ in 1:sm.cardinality
            ]
            argmax(((rfa, satmask),) -> begin
                    sm.quality_evaluator(y[satmask], w[satmask])
                end, randformulas)[1]
        else
            (⊤, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end


############################################################################################
############## Beam search #################################################################
############################################################################################

"""
Beam serch (e.g., CN2)
"""
struct BeamSearch <: SearchMethod end

"""
    function findbestantecedent(
        ::BeamSearch,
        X::PropositionalLogiset,
        y::AbstractVector{<:CLabel},
        w::AbstractVector;
        beam_width::Integer = 3,
        quality_evaluator::Function = soleentropy,
        max_rule_length::Union{Nothing,Integer} = nothing,
        alphabet::Union{Nothing,AbstractAlphabet} = nothing
    )::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

This function performs a beam search to find the best antecedent for a given dataset and labels.
"""
function findbestantecedent(
    ::BeamSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    beam_width::Integer=3,
    quality_evaluator::Function=soleentropy,
    max_rule_length::Union{Nothing,Integer}=nothing,
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

    best = (⊤, ones(Bool, nrow(X)))
    best_quality = quality_evaluator(y, w)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."

    newcandidates = Tuple{RuleAntecedent,SatMask}[]
    while true
        (candidates, newcandidates) = newcandidates, Tuple{RuleAntecedent,SatMask}[]
        #specialize candidate antecedents
        newcandidates = specializeantecedents(candidates, X, max_rule_length, alphabet)
        # Breake if there are no more possible/sensible specializations choices
        isempty(newcandidates) && break

        (perm_, bestcandidate_quality) = sortantecedents(newcandidates, y, w, beam_width, quality_evaluator)
        # (perm_, bestcandidate_quality) = sortantecedents_new(newcandidates, y, beam_width)
        newcandidates = newcandidates[perm_]
        if bestcandidate_quality < best_quality
            best = newcandidates[1]
            best_quality = bestcandidate_quality
        end
    end
    return best
end

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
value of the best one.
"""
function sortantecedents(
    antecedents::AbstractVector{<:Tuple{RuleAntecedent, BitVector}},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    quality_evaluator::Function,
)::Tuple{Vector{Int},<:Real}
    isempty(antecedents) && return [], Inf

    antsquality = map(antd -> begin
            _, satinds = antd
            quality_evaluator(y[satinds], w[satinds])
        end, antecedents)

    newstar_perm = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
    bestantecedent_quality = antsquality[newstar_perm[1]]

    return (newstar_perm, bestantecedent_quality)
end

function increment!(
    v::AbstractVector{<:Integer},
    c::AbstractVector{<:Integer},
    Δ::Int64
)
    for i_c in c
        v[i_c] += Δ
    end
end

# function sortantecedents_new(
#     antecedents::AbstractVector{T},
#     y::AbstractVector{<:CLabel},
#     beam_width::Integer,
# )::Tuple{Vector{Int},<:Real} where {
#     T<:Tuple{RuleAntecedent, BitVector},
# }
#     isempty(antecedents) && return [], Inf
#     n_class = unique(y) |> length

#     distribution = zeros(Int64, n_class)
#     distrib_mask = zeros(Bool, length(y))
#     antsquality = []
#     for _ant in antecedents
#         a_, mask = _ant

#         # antecedent covered examples classes
#         difference_mask = (!).(distrib_mask) .& mask
#         antcoverage = y[difference_mask]
#         @inline increment!(distribution, antcoverage, 1)
#         entropy = begin
#             prob = distribution ./ sum(distribution)
#             -sum(prob .* log2.(prob))
#         end
#         push!(antsquality, entropy)
#         distrib_mask = distrib_mask .| difference_mask
#     end
#     newstar_perm = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
#     bestantecedent_quality = antsquality[newstar_perm[1]]

#     return (newstar_perm, bestantecedent_quality)
# end

function filteralphabet(
    X::PropositionalLogiset,
    alph::AbstractAlphabet,
    antecedent::RuleAntecedent
)::Vector{Tuple{Atom,SatMask}}

    conditions = Atom{ScalarCondition}.(atoms(alph))
    possible_conditions = [(a, check(a, X)) for a in conditions if a ∉ atoms(antecedent)]

    return possible_conditions
end

"""
    function filteralphabetoptimized(
        X::PropositionalLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{RuleAntecedent,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Return every possible atom originating from 'alph' such that, when joined with the antecedent,
it results in a change in the distribution of covered examples.
For optimization purposes, each atom corresponds to a tuple containing the *atom* itself
and the *bitmask* of covered examples
"""
function filteralphabetoptimized(
    X::PropositionalLogiset,
    alph::UnionAlphabet,
    antecedent_info::Tuple{RuleAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}

    antecedent, ant_mask = antecedent_info
    conditions = Atom{ScalarCondition}.(atoms(alph))

    filtered_conditions = [(a, check(a, X)) for a ∈ conditions if a ∉ atoms(antecedent)]
    # Return every atom that, attached to the antecedent, bring a change in the
    # distribution of the examles covered by the antecedent itself.
    return [(a, atom_mask) for (a, atom_mask) ∈ filtered_conditions
            if (ant_mask .& atom_mask) != ant_mask]
end

"""
    newatoms(
        X::PropositionalLogiset,
        antecedent::Tuple{RuleAntecedent, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns a list of all possible conditions (atoms) that can be generated from instances of X
and can further specialize the input antecedent"
"""
function newatoms(
    X::PropositionalLogiset,
    antecedent_info::Tuple{RuleAntecedent,BitVector};
    optimize=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Atom{ScalarCondition},BitVector}}

    (antecedent, satindexes) = antecedent_info
    coveredX = slicedataset(X, satindexes; return_view=true)

    selectedalphabet = !isnothing(alph) ? alph :
                       alphabet(coveredX)
    possibleconditions = optimize ? filteralphabetoptimized(X, selectedalphabet, antecedent_info) :
                         filteralphabet(X, selectedalphabet, antecedent)
    return possibleconditions
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
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{RuleAntecedent,SatMask}}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "aphabet must be finite"

    if isempty(antecedents)

        specializedants = Tuple{RuleAntecedent,SatMask}[]
        selectedalphabet = isnothing(default_alphabet) ? alphabet(X) : default_alphabet

        for (metacond, ths) in grouped_featconditions(selectedalphabet)

            op = test_operator(metacond)
            atomslist = SoleData._atoms((metacond, ths))

            # Optimization
            # The order in which the atomslist is iterated varies based on the comparison
            # operator of the metacondition.
            # (≤) ascending order iteration
            # (≥) descending order iteration
            (isordered(op) && polarity(op)) &&
                (atomslist = Iterators.reverse(atomslist))

            # Contain all the antecedents thats can be generated from the
            # (metacondition, treshold) tuple relative to this iteration.
            metacond_relativeants = Tuple{RuleAntecedent,SatMask}[]

            # Remember that tresholds are sorted !
            cumulative_satmask = zeros(Bool, ninstances(X))

            uncoveredslice = collect(1:ninstances(X))
            for atom in atomslist
                # if uncoveredslice is empty, then all next atoms cover the totality of
                # instances in X. This implies that such atoms have no predicting power.
                isempty(uncoveredslice) && break
                atom_satmask = begin
                    uncoveredX = slicedataset(X, uncoveredslice; return_view=false)
                    check(atom, uncoveredX)
                end
                cumulative_satmask[uncoveredslice] = atom_satmask
                uncoveredslice = uncoveredslice[(!).(atom_satmask)]

                push!(metacond_relativeants, (RuleAntecedent([atom]), cumulative_satmask))
            end
            # before being inserted, the antecedents are rearranged in their original order
            (isordered(op) && polarity(op)) &&
                (metacond_relativeants = Iterators.reverse(metacond_relativeants))

            append!(specializedants, metacond_relativeants)
        end
    else
        specializedants = Tuple{RuleAntecedent,SatMask}[]
        for _ant ∈ antecedents

            # i_conjunctibleatoms refer to all the conditions (Atoms) that can be
            # joined to the i-th antecedent. These are calculated only for the values ​​
            # of the instances already covered by the antecedent.
            conjunctibleatoms = newatoms(X, _ant;
                optimize=true,
                alph=default_alphabet)

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

                push!(specializedants, (antformula, new_antcoverage))
            end
        end
    end
    return specializedants
end

############################################################################################
############## Sequential Covering #########################################################
############################################################################################

"""
    function sequentialcovering(
        X::PropositionalLogiset,
        y::AbstractVector{<:CLabel},
        w::Union{Nothing,AbstractVector{U},Symbol} = default_weights(length(y));
        search_method::SearchMethod=BeamSearch(),
        max_rulebase_length::Union{Nothing,Integer}=nothing,
        suppress_parity_warning::Bool=false,
        kwargs...
    )::DecisionList where {U<:Real}

Learn a decision list on an logiset `X` with labels `y` and weights `w` following
the classic [sequential covering](https://christophm.github.io/interpretable-ml-book/rules.html#sequential-covering) learning scheme.
This involves iteratively learning a single rule, and removing the newly covered instances.

# Arguments
TODO

# Examples
TODO

See also [`TODO`](@ref), [`PropositionalLogiset`](@ref), [`BeamSearch`](@ref).

"""
function sequentialcovering(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{U},Symbol}=default_weights(length(y));
    searchmethod::SearchMethod=BeamSearch(),
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    suppress_parity_warning::Bool=false,
    kwargs...
)::DecisionList where {U<:Real}

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

    (ninstances(X) == 0) && error("Empty trainig set")

    y, labels = y |> maptointeger

    # DEBUG
    # uncoveredslice = collect(1:ninstances(X))

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]
    while true
        bestantecedent, bestantecedent_coverage = findbestantecedent(
            searchmethod,
            uncoveredX,
            uncoveredy,
            uncoveredw;
            kwargs...
        )
        bestantecedent == ⊤ && break

        rule = begin
            justcoveredy = uncoveredy[bestantecedent_coverage]
            justcoveredw = uncoveredw[bestantecedent_coverage]
            consequent = labels[bestguess(justcoveredy, justcoveredw)]
            info_cm = (;
                supporting_labels=collect(justcoveredy),
                supporting_weights=collect(justcoveredw)
            )
            consequent_cm = ConstantModel(consequent, info_cm)
            Rule(bestantecedent, consequent_cm)
        end

        push!(rulebase, rule)

        uncoveredX = slicedataset(uncoveredX, (!).(bestantecedent_coverage); return_view=true)
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

    # !allequal(uncoveredy) && @warn "Remaining classes are not all equal; defaultclass represents the best estimate."

    defaultconsequent = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    return DecisionList(rulebase, defaultconsequent)
end

function sole_cn2(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=BeamSearch(), kwargs...)
end

function sole_rand(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=RandSearch(), kwargs...)
end

#= Int.(values(currentrule_distribution)) =#
# currentrule_distribution = Dict(unique(y) .=> 0)

# for c in coveredy
#     currentrule_distribution[c] += 1
# end
