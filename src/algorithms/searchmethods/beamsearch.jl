
############################################################################################
############## Beam search #################################################################
############################################################################################
"""
Search procedure that explores the solution space selectively, maintaining a restricted set of
partial solutions (the "beam") at each step.

This beam is dynamically updated to include the most promising solutions, allowing for
efficient exploration of the solution space without examining all possibilities.

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

# Arguments

* `beam_width` is the width of the beam, i.e., the maximum number of partial solutions to maintain during the search.
* `quality_evaluator` is the function that assigns a score to each partial solution.
* `max_rule_length` specifies the maximum length allowed for a rule in the search algorithm.
If not specified, the beam will be update until no more possible specializations exist.
* `alphabet` allow the specialization of the antecedent only on a constrained set of conditions.
If not specified, , the entire alphabet originated from X is used.

See also
[`SearchMethod`](@ref),
[`RandSearch`](@ref).
[`specializeantecedents`](@ref).
[`alphabet`](@ref).
"""
struct BeamSearch <: SearchMethod end


# TODO prima riga di documentazione ?????????'
"""
    function filteralphabet(
        X::PropositionalLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{RuleAntecedent,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Return every atom that can be derived from 'alph', except those already in the antecedent.

For optimization purposes, paired with each atom, a bitmask indicating which instances of X
are satisfied is returned.

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

A trivial specialization correspond to an antecedent covering the same instances as its parent
"""
function filteralphabetoptimized(
    X::PropositionalLogiset,
    alph::UnionAlphabet,
    antecedent_info::Tuple{RuleAntecedent,SatMask}
)::Vector{Tuple{Atom,SatMask}}

    antecedent, ant_mask = antecedent_info
    conditions = Atom{ScalarCondition}.(atoms(alph))

    filtered_conditions = [(a, check(a, X)) for a ∈ conditions if a ∉ atoms(antecedent)]
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

Specializes rule *antecedents*.
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

        for (metacond, ths) in map(cha->cha.featcondition, alphabets(selectedalphabet))

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
    ::BeamSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    beam_width::Integer=3,
    quality_evaluator::Function=soleentropy,
    max_rule_length::Union{Nothing,Integer}=nothing,
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
# TODO add min_rule_support parameter
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
