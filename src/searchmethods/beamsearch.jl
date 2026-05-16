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
            alphabet(_X; discretizedomain, y=_y, test_operators=[<, ≥], keep_unique=true) :
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
        alphabet(X; discretizedomain, y, keep_unique = true, test_operators=[<, ≥]) : 
            default_alphabet

    conditions = alphabet2conditions(sm.conjuncts_generation_method, _alphabet, X)
    return [Antecedent([f], mask) for (f, mask) in conditions]
end # TODO: Spostare in core.jl, non ha nulla di specifico che abbia a che fare con beamsearch



"""
    specializeantecedents(
        sm,
        antecedents,
        X,
        y;
        max_rule_length=nothing,
        discretizedomain=false,
        default_alphabet=nothing
    )

Generates all possible specializations from a set of existing antecedents.
For each antecedent, it constructs new specializations by adding a
single condition atom and retains only those that actually change the
dataset coverage.

# Arguments
- `sm::SearchMethod`: the search method used to generate new atoms.
- `antecedents::AbstractVector{Antecedent}`: the antecedents that need to be specialized.
- `X::AbstractLogiset`: the dataset to operate on.
- `y::AbstractVector{<:CLabel}`: the labels associated with the instances of `X`.
- `max_rule_length::Union{Nothing,Integer}`: maximum allowed length for the generated rules.
- `discretizedomain::Bool`: if `true`, uses a discretized domain for alphabet generation.
- `default_alphabet::Union{Nothing,AbstractAlphabet}`: default alphabet to use instead of reconstructing it from `X`.

# Returns
A vector of tuples `(new_antecedent, parent_antecedent)` containing the
generated specializations and their starting antecedent.
"""

 function specializeantecedents(
    sm::SearchMethod,
    antecedents::AbstractVector{Antecedent},
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},

    max_rule_length::Union{Nothing,Integer}=nothing,
    discretizedomain::Bool=false,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
)::Vector{Tuple{Antecedent, Union{Nothing, Antecedent}}}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "alphabet must be finite"

    if isempty(antecedents)
        initial_ants = initialize_antecedents(sm, X, y; discretizedomain, default_alphabet)
        return [(ant, nothing) for ant in initial_ants]
    end

    specializedants = Tuple{Antecedent, Union{Nothing, Antecedent}}[]
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
                new_ant = Antecedent(new_formula, new_mask)

                push!(specializedants, (new_ant, antecedent))
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
    init_best_antecedent(y, w, loss_function; nlabels, kwargs...)

Generates an initial Antecedent "bot" and calcualtes its loss on the dataset.

# Arguments
- `y`: labels vector
- `w`: weights vector
- `loss_function`: the loss function, must accept `(y, w; nlabels)`
- `nlabels`: number of labels (keyword for the loss function)
- `kwargs`: other keyword arguments passed to the loss function

# Returns
A tuple `(best_antecedent, best_loss)`
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
    findbestantecedent(
        bs,
        X,
        y,
        w,
        loss_function,
        max_infogain_ratio,
        default_alphabet,
        discretizedomain,
        significance_alpha,
        min_rule_coverage;
        nlabels,
        max_rule_length=nothing,
        target_class=nothing,
        starting_antecedent=nothing,
        effective_loss=LossFunctions.LaplaceAccuracy(),
        num_features_considered_per_test=nothing,
        rng=Random.default_rng(),
        kwargs...
    )

Finds the best antecedent under beam search using an asymmetric loss function.
This function expands candidate antecedents iteratively, evaluates them with the
provided loss and pruning criteria, and returns the antecedent with the lowest
loss found by the beam search.

# Arguments
- `bs::BeamSearch`: the beam search strategy and beam width.
- `X::AbstractLogiset`: the dataset used to generate and specialize antecedents.
- `y::AbstractVector{<:Integer}`: target labels for the instances in `X`.
- `w::AbstractVector`: instance weights for loss computation.
- `loss_function::LossFunctions.AsymmetricLoss`: the asymmetric loss to optimize.
- `max_infogain_ratio::Union{Real, Nothing}`: maximum information gain ratio allowed for new antecedents.
- `default_alphabet::Union{Nothing,AbstractAlphabet}`: optional predefined alphabet used for condition generation.
- `discretizedomain::Bool`: whether to discretize the domain when building the alphabet.
- `significance_alpha::Real`: significance level used for statistical pruning.
- `min_rule_coverage::Integer`: minimum number of instances a candidate antecedent must cover.
- `nlabels::Integer`: number of label classes used by the loss function.
- `max_rule_length::Union{Integer,Nothing}`: maximum rule length allowed for candidate antecedents.
- `target_class::Union{Integer,Nothing}`: target class for asymmetric losses.
- `starting_antecedent::Union{Nothing, Antecedent}`: optional antecedent from which to start the search.
- `effective_loss::LossFunctions.AsymmetricLoss`: effective loss used when the provided loss is a delta loss.
- `num_features_considered_per_test::Union{Nothing, Integer}`: number of randomly selected features to consider per test.
- `rng::AbstractRNG`: random number generator for feature selection.
- `kwargs...`: additional keyword arguments forwarded to the loss function.

# Returns
The best `Antecedent` found by the beam search.
"""
function findbestantecedent(
    bs::BeamSearch,

    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector,

    loss_function::LossFunctions.AsymmetricLoss,
    max_infogain_ratio::Union{Real, Nothing},
    default_alphabet::Union{Nothing,AbstractAlphabet},
    discretizedomain::Bool,
    significance_alpha::Real,
    min_rule_coverage::Integer;

    nlabels::Integer,
    max_rule_length::Union{Integer,Nothing} = nothing,
    target_class::Union{Integer,Nothing} = nothing,  # this is passed down to the loss function
    starting_antecedent::Union{Nothing, Antecedent} = nothing,

    effective_loss::LossFunctions.AsymmetricLoss = LossFunctions.LaplaceAccuracy(),

    num_features_considered_per_test::Union{Nothing, Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),        # necessary for random feature selection when building tests if num_features_considered_per_test is not equal to nfeatures(X)

    kwargs...
)::Antecedent

    @unpack conjuncts_generation_method, beam_width = bs

    
    # Initializes the best antecedent as the formuala ⊤, unless starting_antecedent is set
    loss_for_starting_candidate = (LossFunctions.is_delta_loss(loss_function)) ? effective_loss : loss_function
    best, best_loss = if isnothing(starting_antecedent)
        init_best_antecedent(y, w, loss_for_starting_candidate; nlabels, target_class = target_class, kwargs...)
    else 
        best_loss = loss_for_starting_candidate(y, w, target_class; antecedent = starting_antecedent, nlabels=nlabels, kwargs...)
        starting_antecedent, best_loss
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
                                                    kwargs...)

        isempty(newcandidates) && break

        newcandidate = newcandidates[begin]     # only keep the best new candidate (in terms of its loss value)

        if LossFunctions.is_delta_loss(loss_function)
            for candidate in newcandidates
                # if a delta loss is passed as an effective loss (for some reason) then we must have prev_antecedent
                abs_loss = effective_loss(y, w, target_class; antecedent=candidate, prev_antecedent=best, nlabels=nlabels, kwargs...)
                if abs_loss < best_loss
                    best_loss = abs_loss
                    best = candidate
                end
            end
        else
            should_update = bestcandidate_loss < best_loss
            should_update && (best = newcandidate; best_loss = bestcandidate_loss)
        end
    end

    return best
end




"""
    findbestantecedent(
        bs,
        X,
        y,
        w,
        loss_function,
        max_infogain_ratio,
        default_alphabet,
        discretizedomain,
        significance_alpha,
        min_rule_coverage;
        nlabels,
        max_rule_length=nothing,
        target_class=nothing,
        starting_antecedent=nothing,
        num_features_considered_per_test=nothing,
        rng=Random.default_rng(),
        kwargs...
    )

Finds the best antecedent under beam search using a symmetric loss function.
This variant does not require a target class and selects the antecedent that
minimizes the symmetric loss across the beam's candidate specializations.

# Arguments
- `bs::BeamSearch`: the beam search strategy and beam width.
- `X::AbstractLogiset`: the dataset used to generate and specialize antecedents.
- `y::AbstractVector{<:Integer}`: target labels for the instances in `X`.
- `w::AbstractVector`: instance weights for loss computation.
- `loss_function::LossFunctions.SymmetricLoss`: the symmetric loss to optimize.
- `max_infogain_ratio::Union{Real, Nothing}`: maximum information gain ratio allowed for new antecedents.
- `default_alphabet::Union{Nothing,AbstractAlphabet}`: optional alphabet used for condition generation.
- `discretizedomain::Bool`: whether to discretize the domain when building the alphabet.
- `significance_alpha::Real`: significance level used for statistical pruning.
- `min_rule_coverage::Integer`: minimum number of instances a candidate antecedent must cover.
- `nlabels::Integer`: number of label classes used by the loss function.
- `max_rule_length::Union{Integer,Nothing}`: maximum rule length allowed for candidate antecedents.
- `target_class::Union{Integer,Nothing}`: included for signature compatibility; ignored by symmetric losses.
- `starting_antecedent::Union{Nothing, Antecedent}`: optional antecedent from which to start the search.
- `num_features_considered_per_test::Union{Nothing, Integer}`: number of randomly selected features to consider per test.
- `rng::AbstractRNG`: random number generator for feature selection.
- `kwargs...`: additional keyword arguments forwarded to the loss function.

# Returns
The best `Antecedent` found by the beam search.
"""

function findbestantecedent(
    bs::BeamSearch,

    X::AbstractLogiset,
    y::AbstractVector{<:Integer},
    w::AbstractVector,

    loss_function::LossFunctions.SymmetricLoss,
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

    # Initializes the best antecedent as the formuala ⊤, unless starting_antecedent is set
    best, best_loss = if isnothing(starting_antecedent)
        init_best_antecedent(y, w, loss_function; nlabels, target_class = target_class, kwargs...)
    else 
        loss_function(y, w; antecedent = starting_antecedent, nlabels=nlabels, kwargs...)
        starting_antecedent, best_loss
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

    newcandidates = Antecedent[]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Antecedent[]

        newcandidates = specializeantecedents(bs,
                                            candidates, X_specialization, y,

                                            max_rule_length,
                                            discretizedomain,
                                            default_alphabet)
        
        # extract the actual antecedents, dump their parents
        newcandidates = [ant for (ant, _) in newcandidates]

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
                                                    kwargs...)

        isempty(newcandidates) && break

        newcandidate = newcandidates[begin]     # only keep the best new candidate (in terms of its loss value)


        if bestcandidate_loss < best_loss
            best = newcandidate
            best_loss = bestcandidate_loss
        end
    end

    return best
end
