using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: isordered, polarity
using Parameters

const BSAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom{ScalarCondition}}

############################################################################################
############## Beam search #################################################################
############################################################################################

"""
Search method to be used in 
[`sequentialcovering`](@ref) that simply returns the best atom as the antecedent. 
"""
struct BestAtom <: SearchMethod

# TODO docu for min_rule_coverage
"""
Search method to be used in 
[`sequentialcovering`](@ref) that explores the solution space selectively,
maintaining a restricted set of partial solutions (the "beam") at each step.

The beam is dynamically updated to include the most promising solutions, allowing for
efficient exploration of the solution space without examining all possibilities.

# Keyword Arguments
* `beam_width::Integer = 3` is the width of the beam, i.e., the maximum number of partial solutions to maintain during the search.
* `quality_evaluator::Function = entropy` is the function that assigns a score to each partial solution.
* `max_rule_length::Union{Nothing,Integer} = nothing` specifies the maximum length allowed for a rule in the search algorithm.
* `min_rule_coverage::Union{Nothing,Integer} = 1` specifies the minimum number of instances covered by each rule.
If not specified, the beam will be update until no more possible specializations exist.
* `alphabet::Union{Nothing,AbstractAlphabet} = nothing` allow the specialization of the antecedent only on a constrained set of conditions.
If not specified, , the entire alphabet originated from X is used.

See also
[`sequentialcovering`](@ref),
[`SearchMethod`](@ref),
[`RandSearch`](@ref),
[`specializeconjunctions`](@ref).
"""
@with_kw struct BeamSearch <: SearchMethod
    beam_width::Integer=3
    quality_evaluator::Function=entropy
    max_rule_length::Union{Nothing,Integer}=nothing
    min_rule_coverage::Union{Integer}=1
    reverse_condorder::Bool=false
    conjuncts_search_method::Bool=BestAtom()
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
end

"""
    function filteralphabet(
        X::PropositionalLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{BSAntecedent,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Return every atom that can be derived from 'alph', except those already in the antecedent.

For optimization purposes each atom is returned paired with its coverage bitmask on X

See also
[`BeamSearch`](@ref).
[`filteralphabetoptimized`](@ref),
[`specializeconjunctions`](@ref).
"""
function filteralphabet(
    X::PropositionalLogiset,
    alph::AbstractAlphabet,
    antecedent::BSAntecedent
)::Vector{Tuple{Atom,SatMask}}

    conditions = Atom{ScalarCondition}.(atoms(alph))
    possible_conditions = [(a, check(a, X)) for a in conditions if a ∉ atoms(antecedent)]

    return possible_conditions
end

"""
    function filteralphabetoptimized(
        X::PropositionalLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{BSAntecedent,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Like filteralphabet but with an additional filtering step ensuring that each atom is not a
trivial specialization for the antecedent.

A trivial specialization correspond to an antecedent covering exactly the same instances as its parent.
"""
function filteralphabetoptimized(
    X::PropositionalLogiset,
    alph::UnionAlphabet,
    antecedent_info::Tuple{BSAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}

    antecedent, ant_mask = antecedent_info
    antecedent_atoms =  atoms(antecedent)
    # possible_atoms = Tuple{Atom,SatMask}[]

    # for univ_scalarcond in alphabets(alph)
    #     atomslist = atoms(univ_scalarcond)
    #     # Atoms in usc conjunctible
    #     usc_conjunctible = Tuple{Atom,SatMask}[]

    #     # Remember that tresholds are sorted !
    #     cumulative_satmask = zeros(Bool, ninstances(X))
    #     prevatom_coveredslice = collect(1:ninstances(X))
    #     for atom in atomslist
    #         isempty(prevatom_coveredslice) && break
    #         atom_satmask = begin
    #             uncoveredX = slicedataset(X, prevatom_coveredslice; return_view=false)
    #             check(atom, uncoveredX)
    #         end
    #         cumulative_satmask[prevatom_coveredslice] = atom_satmask
    #         prevatom_coveredslice = prevatom_coveredslice[atom_satmask]

    #         ((ant_mask .& cumulative_satmask) != ant_mask) & (atom ∉ antecedent_atoms) &&
    #             push!(usc_conjunctible, (atom, cumulative_satmask))
    #     end
    #     append!(possible_atoms, usc_conjunctible)
    # end
    # return possible_atoms
    filtered_conditions = [(a, check(a, X)) for a ∈ atoms(alph)]
    return [(a, atom_mask) for (a, atom_mask) ∈ filtered_conditions
            if ((ant_mask .& atom_mask) != ant_mask) & (a ∉ antecedent_atoms)]
end

"""
    newatoms(
        X::PropositionalLogiset,
        antecedent::Tuple{BSAntecedent, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns the list of all possible conditions (atoms) that can be derived from instances
of X and can further refine the input antecedent.
"""
function newatoms(
    X::PropositionalLogiset,
    antecedent_info::Tuple{BSAntecedent,BitVector};
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

function growconjunctions(
    sm::RandSearch,
    X::PropositionalLogiset,
    antecedents::Union{Nothing,Vector{Tuple{BSAntecedent,SatMask}}},
    max_rule_length::Union{Nothing,Integer}=nothing,
    reverse_condorder::Bool=false, # TODO remove, it only applies to BestAtom; maybe it's an hyperparameter of BestAtom?
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{BSAntecedent,SatMask}}
    if isnothing(antecedents)
        return Iterators.map(c->(BSAntecedent([c]), check(c, X)), searchantecedents(sm, X))
    else
        specializedants = Tuple{BSAntecedent,SatMask}[]
        for (conjunction, antcoverage) ∈ antecedents
            
            for conjunct in searchantecedents(sm, X)
                _cov = check(conjunct, X)
                # new_antcformula = antformula ∧ _atom
                pushconjunct!(conjunct, conjunction)
                new_antcoverage = antcoverage .& _cov

                push!(specializedants, (antformula, new_antcoverage))
            end
        end
        return specializedants
    end
end

function growconjunctions(
    ::BestAtom,
    X::PropositionalLogiset,
    antecedents::Union{Nothing,Vector{Tuple{BSAntecedent,SatMask}}},
    max_rule_length::Union{Nothing,Integer}=nothing,
    reverse_condorder::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{BSAntecedent,SatMask}}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "Alphabet must be finite"
    specializedants = Tuple{BSAntecedent,SatMask}[]

    if isnothing(antecedents)
        selectedalphabet = isnothing(default_alphabet) ? alphabet(X, truerfirst=reverse_condorder) : default_alphabet

        for univ_alph in alphabets(selectedalphabet) # TODO rename atoms.(alphabets(a)) into something like groupedatoms(a)..?

            atomslist = atoms(univ_alph)
            metacond_relativeants = Tuple{BSAntecedent,SatMask}[]

            # Remember that tresholds are sorted !
            cumulative_satmask = zeros(Bool, ninstances(X))

            prevant_coveredslice = collect(1:ninstances(X))
            for atom in atomslist
                isempty(prevant_coveredslice) && break
                atom_satmask = begin
                    uncoveredX = slicedataset(X, prevant_coveredslice; return_view=false) # TODO optimize
                    check(atom, uncoveredX)
                end

                cumulative_satmask[prevant_coveredslice] = atom_satmask
                prevant_coveredslice = prevant_coveredslice[atom_satmask]

                push!(metacond_relativeants, (BSAntecedent([atom]), cumulative_satmask))
            end
            append!(specializedants, metacond_relativeants)
        end
        return specializedants
    else
        for _ant ∈ antecedents

            # i_conjunctibleatoms refer to all the conditions (Atoms) that can be
            # joined to the i-th antecedent. These are calculated only for the values ​​
            # of the instances already covered by the antecedent.
            conjunctibleatoms = newatoms(X, _ant;
                truerfirst=reverse_condorder,
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
        return specializedants
    end
end

"""
    function findbestantecedent(
        ::BeamSearch,
        X::PropositionalLogiset,
        y::AbstractVector{<:CLabel},
        w::AbstractVector
    )::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

Performs a beam search to find the best antecedent for a given dataset and labels.

For further details, please refer to [`BeamSearch`](@ref).
"""
function findbestantecedent(
    bs::BeamSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}


    @unpack beam_width, quality_evaluator, max_rule_length,
        min_rule_coverage, reverse_condorder, alphabet, conjuncts_search_method = bs

    best = (⊤, ones(Bool, nrow(X)))
    best_quality = quality_evaluator(y, w)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."

    newcandidates = nothing
    while true
        # Specialize candidate antecedents
        newcandidates = growconjunctions(conjuncts_search_method, newcandidates, X, max_rule_length, reverse_condorder, alphabet)

        # Break if there are no more possible/sensible specializations
        isempty(newcandidates) && break

        (perm_, bestcandidate_quality) = best_satmasks(last.(newcandidates), y, w, beam_width, quality_evaluator)

        # (perm_, bestcandidate_quality) = best_satmasks_new(newcandidates, y, beam_width)

        newcandidates = newcandidates[perm_]
        
        if bestcandidate_quality < best_quality
            best = newcandidates[1]
            best_quality = bestcandidate_quality
        end
    end
    return best
end
