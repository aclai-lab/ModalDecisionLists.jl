using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: AbstractLogiset
using SoleData: isordered, polarity, metacond
using Parameters
using ModalDecisionLists.Measures: entropy, laplace_accuracy

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
* `loss_function::Function = soleentropy` is the function that assigns a score to each partial solution.
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
@with_kw mutable struct BeamSearch <: SearchMethod
    conjuncts_search_method::SearchMethod=AtomSearch()
    beam_width::Integer=3
    loss_function::Function=entropy
    discretizedomain::Bool=false
    alphabet::Union{Nothing,AbstractAlphabet}=nothing
    max_purity_const::Union{Real,Nothing}=0.0
    significance_alpha::Union{Real,Nothing}=0.0
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
    antecedent::Tuple{Formula,SatMask}
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
function unaryconditions(
    ::AtomSearch,
    a::AbstractAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Atom,SatMask}}
    conditions = Tuple{Atom{ScalarCondition},SatMask}[]
    for univalph in alphabets(a)
        newconds = [(a, check(a, X)) for a in atoms(univalph)]
        append!(conditions, newconds)
    end
    return conditions
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
    ::AtomSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    antecedent::Tuple{Formula,BitVector};
    discretizedomain=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Atom{ScalarCondition},BitVector}}

    antformula, satindexes = antecedent
    coveredX = slicedataset(X, satindexes; return_view=false)
    coveredy = y[satindexes]

    selectedalphabet = begin
        if !isnothing(alph)
            alph
        else
            alphabet(coveredX;
                discretizedomain = discretizedomain,
                y                = coveredy)
        end
    end
    metaconditions = begin
        scalarconditions = value.(children(antformula))
        metacond.(scalarconditions)
    end
    # Exlude metaconditons tha are already in `antecedent`
    selectedalphabet = UnionAlphabet([ a for a in alphabets(selectedalphabet)
            if metacond(a) ∉ metaconditions
        ])
    return filteralphabet(X, selectedalphabet, antecedent)
end


###
prune_noncovering(antecedents) = [a for a in antecedents if ((_, cov) = a; any(cov))]

"""
    specializeantecedents(
        antecedents::Vector{Tuple{RuleAntecbedent,SatMask}},
        X::AbstractLogiset,
        max_rule_length::Union{Nothing,Integer} = nothing,
    )::Vector{Tuple{Formula, SatMask}}

Specialize rule *antecedents*.
"""
function specializeantecedents(
    sm::SearchMethod,
    antecedents::AbstractVector{Tuple{Formula,SatMask}},
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    max_rule_length::Union{Nothing,Integer}=nothing,
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
        specializedants =  map(cond -> begin
                    (formula, satmask) = cond
                    (LeftmostConjunctiveForm([formula]), satmask)
        end, unaryconditions(sm, selectedalphabet, X))
    else
        specializedants = Tuple{LeftmostConjunctiveForm,SatMask}[]
        for currentantecedent ∈ antecedents

            antformula, antcoverage = currentantecedent
            conjunctibleconditions = newconditions(sm, X, y, currentantecedent;
                        alph             = default_alphabet,
                        discretizedomain = discretizedomain)
            isempty(conjunctibleconditions) && continue

            currentant_specialization = [ begin
                newantformula = deepcopy(antformula)
                pushconjunct!(newantformula, newatom)

                (newantformula, antcoverage .& newatom_satmsk)
            end for (newatom, newatom_satmsk) ∈ conjunctibleconditions ]
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
        loss_function::Function = soleentropy,
        max_rule_length::Union{Nothing,Integer} = nothing,
        alphabet::Union{Nothing,AbstractAlphabet} = nothing
    )::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

Performs a beam search to find the best antecedent for a given dataset and labels.

For further details, please refer to [`BeamSearch`](@ref).
"""
function findbestantecedent(
    bs::BeamSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector;
    min_rule_coverage::Integer,
    n_labels::Integer,
    max_rule_length::Union{Integer,Nothing},
)::Tuple{Union{Truth,Formula},SatMask}

    @unpack conjuncts_search_method, beam_width, loss_function,
            discretizedomain, alphabet, max_purity_const, significance_alpha = bs

    best = (⊤, ones(Bool, nrow(X)))
    best_lossfnctn = loss_function(y, w; n_labels = n_labels)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."
    newcandidates = Tuple{Formula,SatMask}[]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Tuple{Formula,SatMask}[]
        # @showlc candidates :red
        newcandidates = specializeantecedents(conjuncts_search_method,
                                            candidates, X, y,
                                            max_rule_length,
                                            discretizedomain,
                                            alphabet)
        # Sort the new candidates
        (newcandidates, bestcandidate_lossfnctn) = sortantecedents(newcandidates,
                                            y, w,
                                            beam_width,
                                            loss_function,
                                            min_rule_coverage,
                                            max_purity_const,
                                            significance_alpha;
                                            n_labels=n_labels)
        isempty(newcandidates) && break

        new_bestcandidate, new_bestcandidate_satmask = newcandidates[begin]
        # Update the best candidate and its lossfnctn
        if (bestcandidate_lossfnctn < best_lossfnctn)
            best = (new_bestcandidate, new_bestcandidate_satmask)
            best_lossfnctn = bestcandidate_lossfnctn
        end
    end
    return best
end

############################################################################################
############################################################################################
############################################################################################

function find_singlerule(
    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector,
    beam_width::Integer,
    # laplace
    target_class::Integer,
    n_labels::Integer,

    discretizedomain::Bool=false,
    max_rule_length::Union{Nothing,Integer}=nothing,
    alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    max_purity_const::Union{Nothing,Real}=nothing
)::Tuple{Union{Truth,Formula},SatMask}

    best = (⊤, ones(Bool, nrow(X)))
    best_quality = laplace_accuracy(y, w; n_labels, target_class)

    newcandidates = Tuple{Formula,SatMask}[]

    while true
        (candidates, newcandidates) = newcandidates, Tuple{Formula,SatMask}[]
        newcandidates = specializeantecedents(candidates,
                            X, y,
                            max_rule_length, discretizedomain, alphabet
                        )
        # In case of DecisionSet learning all the antecedents that do not cover any instances
        # labeled with the target_class are removed.
        newcandidates = [sant for sant in newcandidates if (
                            (_, satmask) = sant;
                            any(y[satmask] .== target_class)
                        )]
        (perm, bestcandidate_quality) = sortantecedents(newcandidates,
                            y, w,
                            beam_width, laplace_accuracy, max_purity_const;
                            target_class=target_class,
                            n_labels=n_labels
                        )

        isempty(perm) && break

        newcandidates = newcandidates[perm]

        if bestcandidate_quality < best_quality
            best = newcandidates[1]
            best_quality = bestcandidate_quality
        end

    end # end while
    return best
end

function find_rules(
    bs::BeamSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector;
    target_class::Integer,
    n_labels::Integer
)::Vector{Rule}

    @unpack beam_width, loss_function, max_rule_length,
        discretizedomain, alphabet, max_purity_const = bs

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."
    Xuncovered = X
    yuncovered = y
    wuncovered = w

    initial_classdistribution = counts(y, n_labels)
    newcandidates = Tuple{RuleAntecedent,SatMask}[]

    bestrules = []
    while true
        bestantecedent = find_singlerule(
                Xuncovered, yuncovered, wuncovered, beam_width,
                # laplace
                target_class, n_labels,
                # general parameters
                discretizedomain, max_rule_length, alphabet
        )
        (bestant_formula, bestant_coverage) = bestantecedent

        # TODO cambiare da target_class(Integer) a CLabel
        newrule = Rule(bestant_formula, ConstantModel(target_class))
        @show newrule
        push!(bestrules, newrule)

        uncovered_slice = begin
            correctclass_coverage = (yuncovered .== target_class) .& bestant_coverage
            (!).(correctclass_coverage)
        end
        Xuncovered = slicedataset(Xuncovered, uncovered_slice; return_view=true)
        yuncovered = @view yuncovered[uncovered_slice]
        wuncovered = @view wuncovered[uncovered_slice]

        !any(yuncovered .== target_class) && break
    end

    return bestrules
end
