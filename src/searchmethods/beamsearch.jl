using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: AbstractLogiset
using SoleData: isordered, polarity, metacond
using SoleLogics: subalphabets
using Parameters
using ModalDecisionLists.LossFunctions: entropy, laplace_accuracy


############################################################################################
############## Beam search #################################################################
############################################################################################

"""
Search method to be used in
[`sequentialcovering`](@ref) that explores the solution space selectively,
maintaining a restricted set of partial solutions (the "beam") at each step.

The beam is dynamically updated to include the most promising solutions, allowing for
efficient exploration of the solution space without examining all possibilities.

# Keyword Arguments
* `conjuncts_search_method::SearchMethod=AtomSearch()`: Defines the heuristic method by which possible conjuncts are generated during the beam search.
* `beam_width::Integer=3` is the width of the beam, i.e., the maximum number of partial solutions to maintain during the search.

See also
[`sequentialcovering`](@ref),
[`SearchMethod`](@ref),
[`AtomSearch`](@ref),
[`RandSearch`](@ref),
[`specializeantecedents`](@ref).
"""
@kwdef mutable struct BeamSearch <: SearchMethod
    conjuncts_search_method::SearchMethod=AtomSearch()
    beam_width::Integer=3
end

# checkedatoms(X, alph) = map(a -> (a, check(a, X)), atoms(alph))
checkedatoms(X, alph) = [(a, check(a, X)) for a ∈ atoms(alph)]

"""
    function filteralphabetoptimized(
        X::AbstractLogiset,
        alph::UnionAlphabet,
        antecedent_info::Tuple{Formula,SatMask}
    )::Vector{Tuple{Atom,SatMask}}

Like filteralphabet but with an additional filtering step ensuring that each atom is not a
trivial specialization for the antecedent.

A trivial specialization correspond to an antecedent covering exactly the same instances as its parent.
"""
#TODO:  @Edo questo va prevenuto appena si generano gli atomi dall'alfabeto (forse)
function filteralphabet(
    X::AbstractLogiset, 
    alph::UnionAlphabet, 
    antecedent::Antecedent
)::Vector{Tuple{Atom,SatMask}}

    isvalid((atom, mask)) = 
        ((antecedent.covmask .& mask) != antecedent.covmask) && 
        (atom ∉ atoms(ant.formula))

    return filter(isvalid, checkedatoms(X, alph))
end

# # Old
# function filteralphabet(
#     X::AbstractLogiset,
#     alph::UnionAlphabet,
#     ant::Antecedent
# )::Vector{Tuple{Atom,SatMask}}
#
#     # antecedent, ant_mask = antecedent
#     f_atoms = atoms(ant.formula)
#     a_atoms = checkedatoms(X, alph)
#
#     return [(a, atom_mask) for (a, atom_mask) ∈ filtered_conditions
#             if ((ant_mask .& atom_mask) != ant_mask) & (a ∉ _atoms)]
# end
#
#
#
"""
Return the list of all possible antecedents containing a single condition from the alphabet.
"""
# function unarycwnditions(
function alphabet2conditions(
    ::AtomSearch,
    a::UnionAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Atom,SatMask}}

    _conditions = Tuple{Atom{ScalarCondition},SatMask}[]

    for univalph in subalphabets(a)
        newconds = [(a, check(a, X)) for a in atoms(univalph)]
        append!(_conditions, newconds)
    end
    return _conditions
end




"""
    newconditions(
        X::AbstractLogiset,
        antecedent::Tuple{Formula, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns the list of all possible conditions (atoms) that can be derived from instances
of X and can further refine the input antecedent.
"""
function newconditions(
    ::AtomSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    ant::Antecedent;

    discretizedomain=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing

)::Vector{Tuple{Atom{ScalarCondition},SatMask}}

    # antformula, satindexes = antecedent
    coveredX = slicedataset(X, ant.covmask; return_view=false)
    coveredy = y[ant.covmask]

    selectedalphabet = begin
        # Exclude metaconditons tha are already in `antecedent`
        metaconditions = metacond.(SoleData.value.(children(antformula)))

        alphabet = something(default_alphabet, 
                alphabet(coveredX; discretizedomain, y = coveredy))

        UnionAlphabet([ a for a in subalphabets(selectedalphabet)
            if metacond(a) ∉ metaconditions
        ])
    end

    return filteralphabet(X, selectedalphabet, antecedent)
end

# TODO: @Edo in inglese
"""
    initial_antecedents(sm, X, y; discretizedomain=false, default_alphabet=nothing)

Genera antecedenti unari a partire dall'alfabeto specificato o da quello
costruito dal logiset `X`.
"""
function init_ants(
    sm::SearchMethod,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel};
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Antecedent}

    _alphabet = something(default_alphabet,
        alphabet(X; discretizedomain=discretizedomain, y=y)
    )
    conditions = alphabet2conditions(sm, _alphabet, X)
    return [Antecedent([f], mask) for (f, mask) in conditions]
end


# TODO @Edo documentazione
function grow_ants(
    sm::SearchMethod,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel};
    antecedents::AbstractVector{Antecedent},
    #
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Antecedent}
    specialized = Antecedent[]

    for antecedent in antecedents
        # Find a set of conjunctible conditions
        conjconds = newconditions(sm, X, y, antecedent;
                                     alph             = default_alphabet,
                                     discretizedomain = discretizedomain,
                                     )

        isempty(conjconds) && continue

        new_ants = [
            let
                new_formula = deepcopy(antecedent.formula)
                pushconjunct!(new_formula, atom)
                Antecedent(new_formula, antecedent.covmask .& mask)
            end
            for (atom, mask) in conjconds
        ]

        append!(specialized, new_ants)
    end

    return specialized
end






###
prune_noncovering(antecedents::AbstractVector{Antecedent}) = [a for a in antecedents if any(a.covmask)]
"""
    specializeantecedents(
        antecedents::Vector{Tuple{RuleAntecbedent,SatMask}},
        X::AbstractLogiset,
        max_rule_length::Union{Nothing,Integer} = nothing,
    )::Vector{Tuple{Formula, SatMask}}

Specialize rule *antecedents*.
"""
# function specializeantecedents(
#     sm::SearchMethod,
#     antecedents::AbstractVector{Antecedent},
#     X::AbstractLogiset,
#     y::AbstractVector{<:CLabel},
#
#     max_rule_length::Union{Nothing,Integer}=nothing,
#     discretizedomain::Bool=false,
#     default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
#
# )::Vector{Antecedent}
#
#     !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "aphabet must be finite"
#
#     if isempty(antecedents)
#
#         # Seleziona alfabeto (genera uno | default)
#         alphabet = isnothing(default_alphabet) ?
#                 alphabet(X; discretizedomain=discretizedomain, y=y) : 
#                 default_alphabet
#
#         _conditions = conditions(sm, alphabet, X)
#         specializedants =  map((f, satmask) ->  Antecedent([f], satmask), 
#                                _conditions)
#     else
#         specializedants = Antecedent[]
#         for antecedent ∈ antecedents
#
#             # antformula, antcoverage = antecedent
#             conjunctibleconditions = newconditions(sm, X, y, antecedent;
#                         alph             = default_alphabet,
#                         discretizedomain = discretizedomain)
#
#             isempty(conjunctibleconditions) && continue
#
#             currentant_specialization = [ begin
#                 newantformula = deepcopy(antformula)
#                 pushconjunct!(newantformula, newatom)
#
#                 (newantformula, antcoverage .& newatom_satmsk)
#             end for (newatom, newatom_satmsk) ∈ conjunctibleconditions ]
#             append!(specializedants, currentant_specialization)
#         end
#     end
#     return prune_noncovering(specializedants)
# end
#
function specializeantecedents(
    sm::SearchMethod,
    antecedents::AbstractVector{Antecedent},
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},

    max_rule_length::Union{Nothing,Integer}=nothing,
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,

)::Vector{Antecedent}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "aphabet must be finite"

    if isempty(antecedents)

        # Seleziona alfabeto (genera uno | default)
        _alphabet = isnothing(default_alphabet) ?
                alphabet(X; discretizedomain=discretizedomain, y=y) : 
                default_alphabet

        _conditions = alphabet2conditions(sm, _alphabet, X)

        # TODO: @Edo capire perchè il codice commentato equivalente al ciclo fon non funziona
        specializedants = Antecedent[]
        for (c, satmask) in _conditions
            push!(specializedants, Antecedent(LeftmostConjunctiveForm([c]), satmask))
        end
        # specializedants =  map((f, satmask) ->  Antecedent(LeftmostConjunctiveForm([f]), satmask), _conditions)
        #   Tentativo non testato: 
        #   
        # specializedants = map(t -> begin
        #     f, satmask = t
        #     Antecedent(LeftmostConjunctiveForm([f]), satmask)
        # end, _conditions)
        #
        #
    else
        specializedants = Antecedent[]
        for antecedent ∈ antecedents

            # antformula, antcoverage = antecedent
            conjunctibleconditions = newconditions(sm, X, y, antecedent;
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
    w::AbstractVector,


    loss_function::Function,
    max_infogain_ratio::Real,
    default_alphabet::Union{Nothing,AbstractAlphabet},
    discretizedomain::Bool,
    significance_alpha::Real,
    min_rule_coverage::Integer;

    nlabels::Integer,
    max_rule_length::Union{Integer,Nothing},
)::Tuple{Union{Truth,Formula},SatMask}

    @unpack conjuncts_search_method, beam_width = bs

    best = (⊤, ones(Bool, nrow(X)))

    best_lossfnctn = loss_function(y, w; nlabels=nlabels)

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."

    newcandidates = Antecedent[]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Antecedent[]

        newcandidates = specializeantecedents(conjuncts_search_method,
                                            candidates, X, y,

                                            max_rule_length,
                                            discretizedomain,
                                            default_alphabet)

        # Sort new candidates
        (newcandidates, bestcandidate_lossfnctn) = sortantecedents(newcandidates,
                                                    y, w, beam_width,
                                                    loss_function,
                                                    min_rule_coverage,
                                                    max_infogain_ratio,
                                                    significance_alpha;
                                                        #
                                                    nlabels=nlabels)
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
    candidates::AbstractVector{<:Tuple{Formula,SatMask}},
    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector,
    beam_width::Integer,
    # laplace
    target_class,
    nlabels,
    # optional positional
    discretizedomain::Bool=false,
    max_rule_length::Union{Nothing,Integer}=nothing,
    alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    max_infogain_ratio::Union{Nothing,Real}=nothing
)::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}

    while true
        (candidates, newcandidates) = newcandidates, Tuple{Formula,SatMask}[]
        newcandidates = specializeantecedents(candidates,
                            X, y,
                            max_rule_length, discretizedomain, alphabet
                        )
        # In case of unordered learning, all the antecedents that do not cover any instances
        # labeled with the target_class must be removed.
        newcandidates = [sant for sant in newcandidates if (
                            (_, satmask) = sant;
                            any(y[satmask] .== target_class)
                        )]
        (perm, bestcandidate_loss) = sortantecedents(newcandidates,
                            y, w,
                            beam_width, laplace_accuracy, max_infogain_ratio;
                            target_class=target_class,
                            nlabels=nlabels
                        )

        isempty(perm) && break
        newcandidates = newcandidates[perm]
        if bestcandidate_loss < best_loss
            best = newcandidates[1]
            best_loss = bestcandidate_loss
        end
    end
    return best
end

############################################################################################
############################################################################################
############################################################################################


function find_rules(
    bs::BeamSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector;
    target_class::Integer,
    nlabels::Integer
)::Vector{Rule}

    @unpack beam_width, loss_function, max_rule_length,
        discretizedomain, alphabet, max_infogain_ratio = bs

    @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."
    Xuncovered = X
    yuncovered = y
    wuncovered = w

    initial_classdistribution = counts(y, nlabels)
    newcandidates = Tuple{Formula,SatMask}[]

    bestrules = []
    while true
        bestantecedent = find_singlerule(
                Xuncovered, yuncovered, wuncovered, beam_width,
                # laplace
                target_class, nlabels,
                # general parameters
                discretizedomain, max_rule_length, alphabet
        )
        (bestant_formula, bestant_coverage) = bestantecedent

        # TODO change target_class::Integer to target_class::CLabel
        newrule = Rule(bestant_formula, ConstantModel(target_class))
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
