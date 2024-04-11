using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: isordered, polarity
using Parameters

############################################################################################
############## Beam search #################################################################
############################################################################################

# TODO docu for min_rule_coverage
"""
Search method to be used in
[`sequentialcovering`](@ref) that explores the solution space selectively,
maintaining a restricted set of partial solutions (the "beam") at each step.

The beam is dynamically updated to include the most promising solutions, allowing for
efficient exploration of the solution space without examining all possibilities.

# Keyword Arguments
* `beam_width::Integer = 3` is the width of the beam, i.e., the maximum number of partial solutions to maintain during the search.
* `quality_evaluator::Function = soleentropy` is the function that assigns a score to each partial solution.
* `max_rule_length::Union{Nothing,Integer} = nothing` specifies the maximum length allowed for a rule in the search algorithm.
* `min_rule_coverage::Union{Nothing,Integer} = 1` specifies the minimum number of instances covered by each rule.
If not specified, the beam will be update until no more possible specializations exist.
* `alphabet::Union{Nothing,AbstractAlphabet} = nothing` allow the specialization of the antecedent only on a constrained set of conditions.
If not specified, , the entire alphabet originated from X is used.

See also
[`sequentialcovering`](@ref),
[`SearchMethod`](@ref),
[`RandSearch`](@ref),
[`specializeantecedents`](@ref).
"""
@with_kw struct BeamSearch <: SearchMethod
    beam_width::Integer=3
    quality_evaluator::Function=soleentropy
    max_rule_length::Union{Nothing,Integer}=nothing
    min_rule_coverage::Union{Integer}=1
    truerfirst::Bool=false
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
end

"""
    function filteralphabet(
        X::PropositionalLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{RuleAntecedent,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Return every atom that can be derived from 'alph', except those already in the antecedent.

For optimization purposes each atom is returned paired with its coverage bitmask on X

See also
[`BeamSearch`](@ref).
[`filteralphabetoptimized`](@ref),
[`specializeantecedents`](@ref).
"""
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

Like filteralphabet but with an additional filtering step ensuring that each atom is not a
trivial specialization for the antecedent.

A trivial specialization correspond to an antecedent covering exactly the same instances as its parent.
"""
function filteralphabetoptimized(
    X::PropositionalLogiset,
    alph::UnionAlphabet,
    antecedent_info::Tuple{RuleAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}

    antecedent, ant_mask = antecedent_info
    antecedent_atoms =  atoms(antecedent)

    filtered_conditions = [(a, check(a, X)) for a ∈ atoms(alph)]
    return [(a, atom_mask) for (a, atom_mask) ∈ filtered_conditions
            if ((ant_mask .& atom_mask) != ant_mask) & (a ∉ antecedent_atoms)]
end

"""
    newatoms(
        X::PropositionalLogiset,
        antecedent::Tuple{RuleAntecedent, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns the list of all possible conditions (atoms) that can be derived from instances
of X and can further refine the input antecedent.
"""
function newatoms(
    X::PropositionalLogiset,
    antecedent_info::Tuple{RuleAntecedent,BitVector};
    optimize=false,
    truerfirst=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Atom{ScalarCondition},BitVector}}
    (antecedent, satindexes) = antecedent_info
    coveredX = slicedataset(X, satindexes; return_view=true)
    selectedalphabet = !isnothing(alph) ? alph :
                       alphabet(coveredX, truerfirst=truerfirst)
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

Specialize rule *antecedents*.
"""
function specializeantecedents(
    antecedents::Vector{Tuple{RuleAntecedent,SatMask}},
    X::PropositionalLogiset,
    max_rule_length::Union{Nothing,Integer}=nothing,
    truerfirst::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{RuleAntecedent,SatMask}}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "aphabet must be finite"
    specializedants = Tuple{RuleAntecedent,SatMask}[]

    if isempty(antecedents)
        selectedalphabet = isnothing(default_alphabet) ? alphabet(X, truerfirst=truerfirst) : default_alphabet
        for univ_alphabet in alphabets(selectedalphabet)
            atomslist = atoms(univ_alphabet)
            metacond_relativeants = Tuple{RuleAntecedent,SatMask}[]

            # Remember that tresholds are sorted !
            cumulative_satmask = zeros(Bool, ninstances(X))
            prevant_coveredslice = collect(1:ninstances(X))
            for atom in atomslist
                isempty(prevant_coveredslice) && break
                atom_satmask = begin
                    uncoveredX = slicedataset(X, prevant_coveredslice; return_view=false)
                    check(atom, uncoveredX)
                end
                cumulative_satmask[prevant_coveredslice] = atom_satmask
                prevant_coveredslice = prevant_coveredslice[atom_satmask]
                push!(metacond_relativeants, (RuleAntecedent([atom]), cumulative_satmask))
            end
            append!(specializedants, metacond_relativeants)
        end
        return specializedants
    else
        for _ant ∈ antecedents
            conjunctibleatoms = newatoms(X, _ant;
                truerfirst=truerfirst,
                optimize=true,
                alph=default_alphabet)
            isempty(conjunctibleatoms) && continue
            for (_atom, _cov) ∈ conjunctibleatoms

                (antformula, antcoverage) = _ant
                if !isnothing(max_rule_length) && nconjuncts(antformula) >= max_rule_length
                    continue
                end
                antformula = deepcopy(antformula)
                pushconjunct!(antformula, _atom)
                push!(specializedants, (antformula, antcoverage .& _cov))
            end
        end
    end
    return specializedants
end

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

Performs a beam search to find the best antecedent for a given dataset and labels.

For further details, please refer to [`BeamSearch`](@ref).
"""
function findbestantecedent(
    bs::BeamSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

    @unpack beam_width, quality_evaluator, max_rule_length,
        min_rule_coverage, truerfirst, alphabet = bs

    best = (⊤, ones(Bool, nrow(X)))
    best_quality = quality_evaluator(y, w)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."

    newcandidates = Tuple{RuleAntecedent,SatMask}[]
    while true
        (candidates, newcandidates) = newcandidates, Tuple{RuleAntecedent,SatMask}[]
        newcandidates = specializeantecedents(candidates,
                                    X,
                                    max_rule_length,
                                    truerfirst,
                                    alphabet)
        isempty(newcandidates) && break
        (perm_, bestcandidate_quality) = sortantecedents(newcandidates, y, w, beam_width, quality_evaluator)

        newcandidates = newcandidates[perm_]
        if bestcandidate_quality < best_quality
            best = newcandidates[1]
            best_quality = bestcandidate_quality
        end
    end
    return best
end
