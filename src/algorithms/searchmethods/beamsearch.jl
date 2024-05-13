using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: AbstractLogiset
using SoleData: isordered, polarity, metacond
using Parameters
using ModalDecisionLists.Measures: entropy

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
    conjuncts_search_method::SearchMethod=AtomSearch()
    #
    beam_width::Integer=3
    quality_evaluator::Function=entropy
    max_rule_length::Union{Nothing,Integer}=nothing
    truerfirst::Bool=false
    discretizedomain::Bool=false
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
    max_purity_const::Union{Real,Nothing}=nothing
end


"""
    function filteralphabetoptimized(
        X::AbstractLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{RuleAntecedent,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Like filteralphabet but with an additional filtering step ensuring that each atom is not a
trivial specialization for the antecedent.

A trivial specialization correspond to an antecedent covering exactly the same instances as its parent.
"""
function filteralphabet(
    X::AbstractLogiset,
    alph::UnionAlphabet,
    antecedent::Tuple{RuleAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}

    antecedent, ant_mask = antecedent
    antecedent_atoms =  atoms(antecedent)

    filtered_conditions = [(a, check(a, X)) for a ∈ atoms(alph)]
    return [(a, atom_mask) for (a, atom_mask) ∈ filtered_conditions
            if ((ant_mask .& atom_mask) != ant_mask) & (a ∉ antecedent_atoms)]
end


"""
Return the list of all possible antecedents containing a single condition from the alphabet.
"""
function unaryantecedents(
    ::AtomSearch,
    a::AbstractAlphabet,
    X::AbstractLogiset
)
    antecedents = Tuple{RuleAntecedent,SatMask}[]
    for univalph in alphabets(a)
        antslist = [(RuleAntecedent([a]), check(a, X)) for a in atoms(univalph)]
        append!(antecedents, antslist)
    end
    return antecedents
end

"""
    newconditions(
        X::AbstractLogiset,
        antecedent::Tuple{RuleAntecedent, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns the list of all possible conditions (atoms) that can be derived from instances
of X and can further refine the input antecedent.
"""
function newconditions(
    ::AtomSearch
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    antecedent_info::Tuple{RuleAntecedent,BitVector};
    truerfirst=false,
    discretizedomain=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Atom{ScalarCondition},BitVector}}

    antecedent, satindexes = antecedent_info
    coveredX = slicedataset(X, satindexes; return_view=false)
    coveredy = y[satindexes]

    selectedalphabet = begin
        if !isnothing(alph)
            alph
        else
            alphabet(coveredX;
                discretizedomain = discretizedomain,
                truerfirst  = truerfirst,
                y           = coveredy
            )
        end
    end
    metaconditions = begin
        scalarconditions = value.(children(antecedent))
        metacond.(scalarconditions)
    end
    # Exlude metaconditons tha are already in `antecedent`
    selectedalphabet = UnionAlphabet([ a for a in alphabets(selectedalphabet)
            if metacond(a) ∉ metaconditions
        ])
    return filteralphabet(X, selectedalphabet, antecedent_info)
end


################## RandSearch ##############################################################
function newconditions(
    ::RandSearch
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    antecedent_info::Tuple{Formula,BitVector};
    truerfirst=false,
    discretizedomain=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Formula,BitVector}}

    antecedent, satindexes = antecedent_info
    coveredX = slicedataset(X, satindexes; return_view=false)
    coveredy = y[satindexes]

    selectedalphabet = begin
        if !isnothing(alph)
            alph
        else
            alphabet(coveredX;
                discretizedomain = discretizedomain,
                truerfirst  = truerfirst,
                y           = coveredy
            )
        end
    end

    # COntinue here......................................
    metaconditions = begin
        scalarconditions = value.(children(antecedent))
        metacond.(scalarconditions)
    end
    # Exlude metaconditons tha are already in `antecedent`
    selectedalphabet = UnionAlphabet([ a for a in alphabets(selectedalphabet)
            if metacond(a) ∉ metaconditions
        ])
    return filteralphabet(X, selectedalphabet, antecedent_info)
end
###
prune_noncovering(antecedents) = [a for a in antecedents if ((_, cov) = a; any(cov))]

"""
    specializeantecedents(
        antecedents::Vector{Tuple{RuleAntecbedent,SatMask}},
        X::AbstractLogiset,
        max_rule_length::Union{Nothing,Integer} = nothing,
    )::Vector{Tuple{RuleAntecedent, SatMask}}

Specialize rule *antecedents*.
"""
function specializeantecedents(
    sm::SearchMethod,
    antecedents::AbstractVector{Tuple{Formula,SatMask}},
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    max_rule_length::Union{Nothing,Integer}=nothing,
    truerfirst::Bool=false,
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{Formula,SatMask}}
    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "aphabet must be finite"

    if isempty(antecedents)
        selectedalphabet = begin
            if isnothing(default_alphabet)
                alphabet(X;
                    discretizedomain = discretizedomain,
                    y = y
                )
            else default_alphabet
            end
        end
        # SearchMethos
        specializedants = unaryantecedents(sm, selectedalphabet, X)
    else
        specializedants = Tuple{Formula,SatMask}[]
        for currentantecedent ∈ antecedents

            antformula,antcoverage = currentantecedent
            conjunctibleatoms = newconditions(sm, X, y, currentantecedent;
                        optimize         = true,
                        truerfirst       = truerfirst,
                        alph             = default_alphabet,
                        discretizedomain = discretizedomain
                    )
            isempty(conjunctibleatoms) && continue

            currentant_specialization = [ begin
                newantformula = deepcopy(antformula)
                pushconjunct!(newantformula, newatom)

                (newantformula, antcoverage .& newatom_satmsk)
            end for (newatom, newatom_satmsk) ∈ conjunctibleatoms ]

            append!(specializedants, currentant_specialization)
        end
    end
    return prune_noncovering(specializedants)
end

function exitcondition(
    candidates,
    max_rule_length::Integer
)
    e = begin
        if isempty(candidates)
            true
        else
            # it is assumed that all antecedents to a given iteration have the same length
            f, _ = candidates[begin]
            if nconjuncts(f) > max_rule_length
                true
            else false
            end
        end
    end
    return e
end

"""
    function findbestantecedent(
        ::BeamSearch,
        X::AbstractLogiset,
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
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    min_rule_coverage::Integer,
    n_labels::Integer
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

    @unpack conjuncts_search_method, beam_width, quality_evaluator, max_rule_length,
            truerfirst, discretizedomain, alphabet, max_purity_const = bs

    best = (⊤, ones(Bool, nrow(X)))
    best_quality = quality_evaluator(y, w; n_labels = n_labels)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."
    newcandidates = Tuple{Formula,SatMask}[]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Tuple{Formula, SatMask}[]
        newcandidates = specializeantecedents(conjuncts_search_method,
                                            candidates, X, y,
                                            max_rule_length,
                                            truerfirst,
                                            discretizedomain,
                                            alphabet)
        # Sort the new candidates
        (newcandidates, bestcandidate_quality) = sortantecedents(newcandidates,
                                            y, w,
                                            beam_width,
                                            quality_evaluator,
                                            min_rule_coverage,
                                            max_purity_const;
                                            n_labels=n_labels)
        isempty(newcandidates) && break
        new_bestcandidate, new_bestcandidate_satmask = newcandidates[begin]
        # Update the best candidate and its quality
        if bestcandidate_quality < best_quality
            best = (new_bestcandidate, new_bestcandidate_satmask)
            best_quality = bestcandidate_quality
        end
    end
    return best
end
