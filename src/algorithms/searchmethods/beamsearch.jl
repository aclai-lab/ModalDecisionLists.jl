using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: AbstractLogiset
using SoleData: isordered, polarity, metacond
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
    quality_evaluator::Function=ModalDecisionLists.Measures.entropy
    max_rule_length::Union{Nothing,Integer}=nothing
    min_rule_coverage::Union{Integer}=1
    truerfirst::Bool=false
    discretizedomain::Bool=false
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
end

"""
    function filteralphabet(
        X::AbstractLogiset,
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
    X::AbstractLogiset,
    alph::AbstractAlphabet,
    antecedent::RuleAntecedent
)::Vector{Tuple{Atom,SatMask}}

    conditions = Atom{ScalarCondition}.(atoms(alph))
    possible_conditions = [(a, check(a, X)) for a in conditions if a ∉ atoms(antecedent)]

    return possible_conditions
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
function filteralphabetoptimized(
    X::AbstractLogiset,
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
        X::AbstractLogiset,
        antecedent::Tuple{RuleAntecedent, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns the list of all possible conditions (atoms) that can be derived from instances
of X and can further refine the input antecedent.
"""
function newatoms(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    antecedent_info::Tuple{RuleAntecedent,BitVector};
    optimize=false,
    truerfirst=false,
    discretizedomain=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Atom{ScalarCondition},BitVector}}
    (antecedent, satindexes) = antecedent_info
    coveredX = slicedataset(X, satindexes; return_view=false)
    coveredy = y[satindexes]
    selectedalphabet = !isnothing(alph) ? alph :
                       alphabet(coveredX;
                            truerfirst=truerfirst,
                            discretizedomain=discretizedomain,
                            y=coveredy)

    scalarconditions = value.(children(antecedent))
    metaconditions = metacond.(scalarconditions)

    # Exlude metaconditons tha are already in `antecedent`
    selectedalphabet = [ a for a in alphabets(selectedalphabet)
                                if metacond(a) ∉ metaconditions
                        ] |> UnionAlphabet

    possibleconditions = optimize ? filteralphabetoptimized(X, selectedalphabet, antecedent_info) :
                        filteralphabet(X, selectedalphabet, antecedent)
    return possibleconditions
end

# Siamo sicuri che la vogliamo cosi ????
# Pensa a quando verrà parallelizzata
function univariate_unaryantecedents(
    X::AbstractLogiset,
    univ_alphabet::AbstractAlphabet
)
    atomslist = atoms(univ_alphabet)
    antdslist = Tuple{RuleAntecedent,SatMask}[]

    cumulativemask = zeros(Bool, ninstances(X))
    prevant_coverage = collect(1:ninstances(X))

    for atom in atomslist
        isempty(prevant_coverage) && break
        atom_satmask = begin
            uncoveredX = slicedataset(X, prevant_coverage; return_view=false)
            check(atom, uncoveredX)
        end
        cumulativemask[prevant_coverage] = atom_satmask
        prevant_coverage = prevant_coverage[atom_satmask]
        push!(antdslist, (RuleAntecedent([atom]), cumulativemask))
    end
    return antdslist
end


"""
    specializeantecedents(
        antecedents::Vector{Tuple{RuleAntecbedent,SatMask}},
        X::AbstractLogiset,
        max_rule_length::Union{Nothing,Integer} = nothing,
    )::Vector{Tuple{RuleAntecedent, SatMask}}

Specialize rule *antecedents*.
"""
function specializeantecedents(
    antecedents::Vector{Tuple{RuleAntecedent,SatMask}},
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    max_rule_length::Union{Nothing,Integer}=nothing,
    truerfirst::Bool=false,
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{RuleAntecedent,SatMask}}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "aphabet must be finite"

    specializedants = Tuple{RuleAntecedent,SatMask}[]

    if isempty(antecedents)
        selectedalphabet = isnothing(default_alphabet) ? alphabet(
                                                    X;
                                                    truerfirst = truerfirst,
                                                    discretizedomain=discretizedomain,
                                                    y=y) :
                                                        #= Or =#
                                                    default_alphabet
        for univ_alphabet in alphabets(selectedalphabet)
            univariate_ants = univariate_unaryantecedents(X, univ_alphabet)
            append!(specializedants, univariate_ants)
        end
    else
        for _ant ∈ antecedents
            conjunctibleatoms = newatoms(X,y, _ant;
                optimize=true,
                truerfirst=truerfirst,
                alph=default_alphabet,
                discretizedomain=discretizedomain,)
            #
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
    w::AbstractVector,
    n_labels::Integer;
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

    @unpack beam_width, quality_evaluator, max_rule_length,
        min_rule_coverage, truerfirst, discretizedomain, alphabet = bs

    best = (⊤, ones(Bool, nrow(X)))
    best_quality = quality_evaluator(y, w; n_labels)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."

    newcandidates = Tuple{RuleAntecedent,SatMask}[]
    while true
        (candidates, newcandidates) = newcandidates, Tuple{RuleAntecedent,SatMask}[]
        newcandidates = specializeantecedents(candidates,
                                    X,y,
                                    max_rule_length,
                                    truerfirst,
                                    discretizedomain,
                                    alphabet)
        isempty(newcandidates) && break
        (perm_, bestcandidate_quality) = sortantecedents(newcandidates, y, w, beam_width, quality_evaluator; n_labels)

        newcandidates = newcandidates[perm_]
        if bestcandidate_quality < best_quality
            best = newcandidates[1]
            best_quality = bestcandidate_quality
        end
    end
    return best
end
