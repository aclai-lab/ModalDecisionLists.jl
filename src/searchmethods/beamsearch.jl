using SoleLogics: AbstractAlphabet, pushconjunct!
using SoleData: AbstractLogiset
using SoleData
using SoleData: isordered, polarity, metacond, features
using SoleLogics: subalphabets
using Parameters
using StatsBase
using Random
using DataFrames
using Tables
using .LossFunctions

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
* `conjuncts_generation_method::SearchMethod=AtomSearch()`: Defines the heuristic method by which possible conjuncts are generated during the beam search.
* `beam_width::Integer=3` is the width of the beam, i.e., the maximum number of partial solutions to maintain during the search.

See also
[`sequentialcovering`](@ref),
[`SearchMethod`](@ref),
[`AtomSearch`](@ref),
[`RandSearch`](@ref),
[`specializeantecedents`](@ref).
"""
mutable struct BeamSearch <: SearchMethod
    # conjuncts_generation_method::AbstractGenerator=AtomGenerator()
    conjuncts_generation_method::AbstractGenerator
    beam_width::Integer

    function BeamSearch(conjuncts_generation_method::AbstractGenerator=AtomGenerator(), beam_width::Integer=3)
        if beam_width < 1
            throw(ArgumentError("`beam_width` must be ≥ 1, got $beam_width"))
        end
        new(conjuncts_generation_method, beam_width)
    end
end

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
function filterconditions(
    conditions::Vector{Tuple{Atom,SatMask}},
    ant::Antecedent,
)::Vector{Tuple{Atom,SatMask}}

    existing_atoms = Set(atoms(ant.formula))

    # An atom is considered active for a given antecedent iff its addition 
    # changes the set of covered instances in the dataset.
    is_active((atom, mask)) = 
        ((ant.covmask .& mask) != ant.covmask) && 
             (atom ∉ existing_atoms)

    return filter(is_active, conditions)
end





"""
Return the list of all possible antecedents containing a single condition from the alphabet.
"""
metaconds(a::Antecedent) = metacond.(SoleData.value.(children(a.formula)))

"""
    newconditions(
        X::AbstractLogiset,
        antecedent::Tuple{Formula, SatMask}
    )::Vector{Tuple{Atom, SatMask}}

Returns the list of all possible conditions (atoms) that can be derived from instances
of X and can further refine the input antecedent.\\
Only refinements that do not cover exactly the same samples as the original antecedent 'ant' are returned.
"""
function newconditions(
    sm::SearchMethod,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    ant::Antecedent;

    discretizedomain=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing
)::Vector{Tuple{Atom{ScalarCondition},SatMask}}

    # dataset composed of the samples covered by 'ant'
    _X = slicedataset(X, ant.covmask; return_view=false)
    _y = y[ant.covmask]

    selectedalphabet = begin
        # make sure to create the alphabet automatically if default_alphabet is null
        _alphabet = isnothing(default_alphabet) ? 
            # alphabet(_X; discretizedomain, y=_y, sortingmode = :generalfirst) :
            alphabet(_X; discretizedomain, y=_y, test_operators=[<, ≥]) :
            default_alphabet

        UnionAlphabet([_alphabet])   # return is cleaner
    end
    
    conditions = alphabet2conditions(sm.conjuncts_generation_method, selectedalphabet, X)
    return filterconditions(conditions, ant)
end


"""
    initial_antecedents(sm, X, y; discretizedomain=false, default_alphabet=nothing)

Generates a list of unary antecedents starting from the specified alphabet, or from that
built from the logiset `X`. The antecedents are built using the specified search method's
procedure 'conjuncts_generation_method'
"""
function initialize_antecedents(
    sm::SearchMethod, 
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel};
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Antecedent}

    _alphabet = isnothing(default_alphabet) ?
        # alphabet(X; discretizedomain, y, sortingmode = :generalfirst) :
        alphabet(X; discretizedomain, y, test_operators=[<, ≥]) : 
            default_alphabet

    conditions = alphabet2conditions(sm.conjuncts_generation_method, _alphabet, X)
    return [Antecedent([f], mask) for (f, mask) in conditions]
end # TODO: Spostare in core.jl, non ha nulla di specifico che abbia a che fare con beamsearch


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
    antecedents::AbstractVector{Antecedent},
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},

    max_rule_length::Union{Nothing,Integer}=nothing,
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Antecedent}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "alphabet must be finite"

    if isempty(antecedents)
        return initialize_antecedents(sm, X, y; discretizedomain, default_alphabet)
    end

    specializedants = Antecedent[]
    # pre-allocate based on an assumption of branching factor: we can assume an average of two specializations for each current antecedent
    sizehint!(specializedants, length(antecedents) * 2)

    for antecedent ∈ antecedents
        conjconds = newconditions(sm, X, y, antecedent; 
                                 discretizedomain=discretizedomain, 
                                 default_alphabet=default_alphabet)

        isempty(conjconds) && continue

        # pull out the existing conjuncts only once per antecedent
        old_conjuncts = children(antecedent.formula)

        for (atom, mask) ∈ conjconds
            # calculate mask first to avoid unnecessary object creation
            new_mask = antecedent.covmask .& mask
            
            # only build the formula if the rule actually covers something
            if any(new_mask)
                # create a new list of conjuncts by copying the old one and adding the new atom.
                new_conjuncts = vcat(old_conjuncts, atom)
                
                # reconstruct the formula without deepcopying the atoms themselves
                new_formula = LeftmostConjunctiveForm(new_conjuncts)
                
                push!(specializedants, Antecedent(new_formula, new_mask))
            end
        end
    end
    
    return specializedants
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
    init_best_antecedent(y, w, loss_function; nlabels)

Crea un Antecedent iniziale "bot" e ne calcola la loss sul dataset.

# Argomenti
- `y`: vettore di etichette
- `w`: vettore di pesi
- `loss_function`: funzione di loss, deve accettare `(y, w; nlabels)`
- `nlabels`: numero di label (keyword per loss_function)

# Ritorna
Una tupla `(best_antecedent, best_loss)`
"""
function init_best_antecedent(y, w, loss_function::LossFunctions.AbstractLossFunction; nlabels, kwargs...)
    return error("Cannot call init_best_antecedent with an AbstractLossFunction type")
end

# For symmetric losses
function init_best_antecedent(y, w, loss_function::LossFunctions.SymmetricLoss; nlabels, kwargs...)
    antecedent = bot_antecedent(length(y))
    loss_val = loss_function(y, w; antecedent=antecedent, nlabels=nlabels, kwargs...)
    return antecedent, loss_val 
end

# For asymmetric losses (the "target_class" attribute must be passed)
function init_best_antecedent(y, w, loss_function::LossFunctions.AsymmetricLoss; nlabels, target_class::Union{Integer,Nothing}=nothing, kwargs...)
    if isnothing(target_class)
        return error("If init_best_antecedent is called with an AsymmetricLoss function, the attribute target_class must be specified")
    end 

    antecedent = bot_antecedent(length(y))
    loss_val = loss_function(y, w, target_class; antecedent=antecedent, nlabels=nlabels, kwargs...)
    return antecedent, loss_val 
end

"""
    function findbestantecedent(
        bs::BeamSearch,

        X::AbstractLogiset,
        y::AbstractVector{<:Integer},
        w::AbstractVector,

        loss_function::LossFunctions.AbstractLossFunction,
        max_infogain_ratio::Union{Real, Nothing},
        default_alphabet::Union{Nothing,AbstractAlphabet},
        discretizedomain::Bool,
        significance_alpha::Real,
        min_rule_coverage::Integer;

        nlabels::Integer,
        max_rule_length::Union{Integer,Nothing} = nothing,
        target_class::Union{Integer,Nothing} = nothing,
        starting_antecedent::Union{Nothing, Antecedent} = nothing,

        num_features_considered_per_test::Union{Nothing, Integer} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )::Antecedent

Performs a beam search to find the best antecedent for a given dataset and labels.
If `num_features_considered_per_test` is passed, a subset of the same number of features is randomly selected, and only those features are used to construct the rule. 

# Note
If num_features_considered_per_test is passed, X must support column-wise indexing such as `X_sub = X[:, my_feats]`

For further details, please refer to [`BeamSearch`](@ref).
"""
function findbestantecedent(
    bs::BeamSearch,

    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector,

    loss_function::LossFunctions.AbstractLossFunction,
    max_infogain_ratio::Union{Real, Nothing},
    default_alphabet::Union{Nothing,AbstractAlphabet},
    discretizedomain::Bool,
    significance_alpha::Real,
    min_rule_coverage::Integer;

    nlabels::Integer,
    max_rule_length::Union{Integer,Nothing} = nothing,
    target_class::Union{Integer,Nothing} = nothing,  # this is passed down to the loss function
    starting_antecedent::Union{Nothing, Antecedent} = nothing,

    num_features_considered_per_test::Union{Nothing, Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),        # necessary for random feature selection when building tests if num_features_considered_per_test is not equal to nfeatures(X)

    kwargs...
)::Antecedent

    @unpack conjuncts_generation_method, beam_width = bs

    # Initializes the best antecedent as the formuala ⊤ 
    if isnothing(starting_antecedent)
        best, best_loss = init_best_antecedent(y, w, loss_function; nlabels, target_class = target_class, kwargs...)
    else
        best = starting_antecedent
        if isa(loss_function, LossFunctions.AsymmetricLoss)
            best_loss = loss_function(y, w, target_class; antecedent = starting_antecedent, nlabels=nlabels, kwargs...)
        else
            best_loss = loss_function(y, w; antecedent = starting_antecedent, nlabels=nlabels, kwargs...)
        end
    end

    # Selects the features if 'num_features_considered_per_test' is specified and not equal to the number of features
    X_specialization = if isnothing(num_features_considered_per_test) || num_features_considered_per_test == nfeatures(X)
        X
    else
        all_feats = collect(Tables.columnnames(Tables.columns(X)))                   # list of feature names, this requires X to be a PropositionalLogiset supporting DataFrame indexing
        # all_feats = features(X)
        selected_features = shuffle(rng, all_feats)[1 : num_features_considered_per_test]   # extract features to be used in the test
        X[:, selected_features]
    end

    # default_alphabet = get_no_nil(default_alphabet, alphabet(X_specialization; test_operators=[<, ≥]))

    newcandidates = Antecedent[]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Antecedent[]

        newcandidates = specializeantecedents(bs,
                                            candidates, X_specialization, y,

                                            max_rule_length,
                                            discretizedomain,
                                            default_alphabet)
        
        # @show newcandidates
        # readline()
        # Sort new candidates
        (newcandidates, bestcandidate_loss) = sortantecedents(newcandidates,
                                                    y, w, beam_width,
                                                    loss_function,
                                                    min_rule_coverage,
                                                    max_infogain_ratio,
                                                    significance_alpha;
                                                        # kwargs vari per tutte le possibili loss functions
                                                    nlabels=nlabels,
                                                    target_class=target_class,
                                                    prev_antecedent=best,
                                                    kwargs...)

        isempty(newcandidates) && break

        newcandidate = newcandidates[begin]     # only keep the best new candidate (in terms of its loss value)

        # Update the best candidate and its lossfnctn
        if (bestcandidate_loss < best_loss)
            best_loss = bestcandidate_loss
            best = newcandidate
        end
    end

    return best
end

############################################################################################
############################################################################################
############################################################################################

# function find_singlerule(
#     candidates::AbstractVector{<:Tuple{Formula,SatMask}},
#     X::AbstractLogiset,
#     y::AbstractVector{<:Integer},
#     w::AbstractVector,
#     beam_width::Integer,
#     # laplace
#     target_class,
#     nlabels,
#     # optional positional
#     discretizedomain::Bool=false,
#     max_rule_length::Union{Nothing,Integer}=nothing,
#     alphabet::Union{Nothing,AbstractAlphabet}=nothing,
#     max_infogain_ratio::Union{Nothing,Real}=nothing
# )::Tuple{Union{Truth,LeftmostConjunctiveForm},SatMask}
#
#     while true
#         (candidates, newcandidates) = newcandidates, Tuple{Formula,SatMask}[]
#         newcandidates = specializeantecedents(candidates,
#                             X, y,
#                             max_rule_length, discretizedomain, alphabet
#                         )
#         # In case of unordered learning, all the antecedents that do not cover any instances
#         # labeled with the target_class must be removed.
#         newcandidates = [sant for sant in newcandidates if (
#                             (_, satmask) = sant;
#                             any(y[satmask] .== target_class)
#                         )]
#         (perm, bestcandidate_loss) = sortantecedents(newcandidates,
#                             y, w,
#                             beam_width, laplace_accuracy, max_infogain_ratio;
#                             target_class=target_class,
#                             nlabels=nlabels
#                         )
#
#         isempty(perm) && break
#         newcandidates = newcandidates[perm]
#         if bestcandidate_loss < best_loss
#             best = newcandidates[1]
#             best_loss = bestcandidate_loss
#         end
#     end
#     return best
# end

############################################################################################,
############################################################################################
############################################################################################


# function find_rules(
#     bs::BeamSearch,
#     X::AbstractLogiset,
#     y::AbstractVector{<:Integer},
#     w::AbstractVector;
#     target_class::Integer,
#     nlabels::Integer
# )::Vector{Rule}
#
#     @unpack beam_width, loss_function, max_rule_length,
#         discretizedomain, alphabet, max_infogain_ratio = bs
#
#     @assert beam_width > 0 "parameter 'beam_width' cannot be less than one. Please provide a valid value."
#     !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
#                                                                "than one. Please provide a valid value."
#     Xuncovered = X
#     yuncovered = y
#     wuncovered = w
#
#     initial_classdistribution = counts(y, nlabels)
#     newcandidates = Tuple{Formula,SatMask}[]
#
#     bestrules = []
#     while true
#         bestantecedent = find_singlerule(
#                 Xuncovered, yuncovered, wuncovered, beam_width,
#                 # laplace
#                 target_class, nlabels,
#                 # general parameters
#                 discretizedomain, max_rule_length, alphabet
#         )
#         (bestant_formula, bestant_coverage) = bestantecedent
#
#         # TODO change target_class::Integer to target_class::CLabel
#         newrule = Rule(bestant_formula, ConstantModel(target_class))
#         push!(bestrules, newrule)
#
#         uncovered_slice = begin
#             correctclass_coverage = (yuncovered .== target_class) .& bestant_coverage
#             (!).(correctclass_coverage)
#         end
#         Xuncovered = slicedataset(Xuncovered, uncovered_slice; return_view=true)
#         yuncovered = @view yuncovered[uncovered_slice]
#         wuncovered = @view wuncovered[uncovered_slice]
#
#         !any(yuncovered .== target_class) && break
#     end
#
#     return bestrules
# end
