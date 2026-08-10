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
        ((ant.covmask .& mask) != ant.covmask) && (atom ∉ existing_atoms)

    return filter(is_active, conditions)
end


custom_ops(::Type{<:Number}) = [<, ≥]
custom_ops(::Type{<:Any}) = [(==), (≠)]

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
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    precomputed_conditions::Union{Nothing,Vector{Tuple{Atom, BitVector}}}=nothing
)::Vector{Tuple{Atom{ScalarCondition},SatMask}}

    if !isnothing(precomputed_conditions)
        return filterconditions(precomputed_conditions, ant)
    end

    # dataset composed of the samples covered by 'ant'
    _X = slicedataset(X, ant.covmask; return_view=true)
    _y = y[ant.covmask]

    selectedalphabet = begin
        # make sure to create the alphabet automatically if default_alphabet is null
        _alphabet = isnothing(default_alphabet) ? 
            alphabet(_X; discretizedomain, y=_y, keep_unique=true, test_operators = [<, ≥]) :
            default_alphabet

        UnionAlphabet([_alphabet])
    end
    
    conditions = alphabet2conditions(sm.conjuncts_generation_method, selectedalphabet, X, discretizedomain)
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
    precomputed_conditions::Union{Nothing, Vector{Tuple{Atom, BitVector}}}=nothing
)::Vector{Antecedent}

    _alphabet = isnothing(default_alphabet) ?
        alphabet(X; discretizedomain, y, keep_unique = true, test_operators = [<, ≥]) : 
            default_alphabet

    conditions = isnothing(precomputed_conditions) ? alphabet2conditions(sm.conjuncts_generation_method, _alphabet, X, discretizedomain) : precomputed_conditions
    return [Antecedent([f], mask) for (f, mask) in conditions]
end 



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
    precomputed_conditions::Union{Nothing, Vector{Tuple{Atom, BitVector}}}=nothing
)::Vector{Tuple{Antecedent, Union{Nothing, Antecedent}}}

    !isnothing(default_alphabet) && @assert isfinite(default_alphabet) "alphabet must be finite"

    if isempty(antecedents)
        initial_ants = initialize_antecedents(sm, X, y; discretizedomain, default_alphabet, precomputed_conditions)
        return [(ant, nothing) for ant in initial_ants]
    end

    specializedants = Tuple{Antecedent, Union{Nothing, Antecedent}}[]
    # pre-allocate based on an assumption of branching factor: we can assume an average of two specializations for each current antecedent
    sizehint!(specializedants, length(antecedents) * 2)

    for antecedent ∈ antecedents
        conjconds = newconditions(sm, X, y, antecedent; 
                                 discretizedomain=discretizedomain, 
                                 default_alphabet=default_alphabet,
                                 precomputed_conditions=precomputed_conditions)

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
function init_best_antecedent(
    y, 
    w, 
    loss_function::LossFunctions.SymmetricLoss; 
    starting_antecedent::Union{Nothing, Antecedent} = nothing,
    nlabels, 
    kwargs...
)
    if isnothing(starting_antecedent)
        antecedent = bot_antecedent(length(y))
        loss_val = loss_function(y, w; antecedent=antecedent, nlabels, kwargs...)
        return antecedent, loss_val
    end


    best_loss = loss_function(y, w; antecedent = starting_antecedent, nlabels, kwargs...)
    return starting_antecedent, best_loss
end

# For asymmetric losses (the "target_class" attribute must be passed)
function init_best_antecedent(
    y, 
    w, 
    loss_function::LossFunctions.AsymmetricLoss; 
    nlabels, 
    target_class::Union{Integer,Nothing}=nothing, 
    starting_antecedent::Union{Nothing, Antecedent} = nothing,
    kwargs...
)
    if isnothing(target_class)
        return error("If init_best_antecedent is called with an AsymmetricLoss function, the attribute target_class must be specified")
    end 

    if isnothing(starting_antecedent)
        antecedent = bot_antecedent(length(y))
        loss_val = loss_function(y, w, target_class; antecedent=antecedent, nlabels, kwargs...)
        return antecedent, loss_val
    end

    best_loss = loss_function(y, w, target_class; antecedent = starting_antecedent, nlabels, kwargs...)

    return starting_antecedent, best_loss
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
- `feature_selection_strategy`: strategy used to select which features are considered when generating candidate conditions; defaults to `DefaultFeatureSelector()`.
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
    rng::AbstractRNG = Random.default_rng(),        # necessary for random feature selection when building tests if num_features_considered_per_test is not equal to nfeatures(X)

    feature_selection_strategy = DefaultFeatureSelector(),

    kwargs...
)::Antecedent

    @unpack conjuncts_generation_method, beam_width = bs

    isnothing(default_alphabet) && (default_alphabet = alphabet(X; discretizedomain, y, keep_unique = true, test_operators = [<, ≥])) 

    precomputed_conditions = if !isnothing(default_alphabet)        # vector of (Atom{ScalarCondition}, BitVector)
        alphabet2conditions(bs.conjuncts_generation_method, UnionAlphabet([default_alphabet]), X, discretizedomain)
    else
        nothing
    end

    # Initializes the best antecedent as the formuala ⊤, unless starting_antecedent is set
    loss_for_starting_candidate = (LossFunctions.is_delta_loss(loss_function)) ? effective_loss : loss_function
    best, best_loss = init_best_antecedent(y, w, loss_for_starting_candidate; nlabels, target_class, starting_antecedent, kwargs...)

    dataset_features = collect(Symbol, Tables.columnnames(Tables.columns(X)))

    newcandidates = isnothing(starting_antecedent) ? Antecedent[] : Antecedent[starting_antecedent]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Antecedent[]

        # select the relevant features for this test using the selection strategy 'feature_selection_strategy'
        selected_features = selectfeatures!(feature_selection_strategy, dataset_features, rng)
        relevant_precomputed_conds = extract_conditions(precomputed_conditions, selected_features, dataset_features)

        X_specialized = (selected_features == dataset_features) ? X : X[:, selected_features]       # avoid making a copy if all the features have been selected

        newcandidates = specializeantecedents(bs,
                                            candidates, X_specialized, y,

                                            max_rule_length,
                                            discretizedomain,
                                            default_alphabet,
                                            relevant_precomputed_conds)
        


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
- `feature_selection_strategy`: strategy used to select which features are considered when generating candidate conditions; defaults to `DefaultFeatureSelector()`.
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

    rng::AbstractRNG = Random.default_rng(),        # necessary for random feature selection when building tests if num_features_considered_per_test is not equal to nfeatures(X)

    feature_selection_strategy = DefaultFeatureSelector(),

    kwargs...
)::Antecedent

    @unpack conjuncts_generation_method, beam_width = bs

    isnothing(default_alphabet) && (default_alphabet = alphabet(X; discretizedomain, y, keep_unique = true, test_operators = [<, ≥])) 

    precomputed_conditions = if !isnothing(default_alphabet)
        alphabet2conditions(bs.conjuncts_generation_method, UnionAlphabet([default_alphabet]), X, discretizedomain)
    else
        nothing
    end

    # Initializes the best antecedent as the formuala ⊤, unless starting_antecedent is set
    best, best_loss = init_best_antecedent(y, w, loss_function; starting_antecedent, nlabels, kwargs...)

    dataset_features = collect(Symbol, Tables.columnnames(Tables.columns(X)))

    newcandidates = isnothing(starting_antecedent) ? Antecedent[] : Antecedent[starting_antecedent]
    while true
        # Generate new specialized candidates
        (candidates, newcandidates) = newcandidates, Antecedent[]

        # select the relevant features for this test using the selection strategy 'feature_selection_strategy'
        selected_features = selectfeatures!(feature_selection_strategy, dataset_features, rng)
        relevant_precomputed_conds = extract_conditions(precomputed_conditions, selected_features, dataset_features)

        X_specialized = (selected_features == dataset_features) ? X : X[:, selected_features]

        newcandidates = specializeantecedents(bs,
                                            candidates, X_specialized, y,

                                            max_rule_length,
                                            discretizedomain,
                                            default_alphabet,
                                            relevant_precomputed_conds)
        
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
