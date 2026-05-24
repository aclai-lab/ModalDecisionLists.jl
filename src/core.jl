using DataFrames

using SoleLogics
using SoleData
using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase
using Random


const SatMask = BitVector


############################################################################################
############ Utilities #####################################################################
############################################################################################



struct Antecedent
    formula::LeftmostConjunctiveForm
    covmask::SatMask
end

function Antecedent(fs::AbstractVector{<:Formula}, cm::SatMask)
    return Antecedent(LeftmostConjunctiveForm(fs), cm)
end

# Funzione per creare un Antecedent "top"
function bot_antecedent(n::Integer)
    return Antecedent(LeftmostConjunctiveForm([⊤]), ones(Bool, n))   # ⊤ rappresenta la formula top 
end

istop(a::Antecedent)  = a.formula.grandchildren == [⊤]

conds(a::Antecedent) = a.formula.grandchildren
nconds(a::Antecedent) = length(conds(a))


function extract_covered_labels(
    ant::Union{Nothing, Antecedent},
    y::AbstractVector{<:CLabel},
    w::Union{Nothing, AbstractVector{<:Real}}
)
    y_covered = if (isnothing(ant) || isempty(ant.covmask))
        Int64[]
    else
        @view y[ant.covmask]
    end

    w_covered = if isnothing(w)
        nothing
    elseif (isnothing(ant) || isempty(ant.covmask))
        Int64[]
    else
        @view w[ant.covmask]
    end

    return y_covered, w_covered
end

# Utilizzare questo Wrapper di Accessors
# @forward Antecedent.formula (
#     SoleLogics.check,  # check(antecedent, X) diventa check(antecedent.formula, X)
#     Base.length,
#     nconjuncts,
#     # ... altri metodi
# )
#
# # Ora puoi fare:
# ant = Antecedent(formula, mask)
# check(ant, X)  # Automaticamente delegato a check(ant.formula, X)
#




struct TrainingState
    X :: AbstractLogiset
    y :: AbstractVector{<:CLabel}
    w :: Union{Nothing,AbstractVector{<:Real}}
    original_y :: Union{AbstractVector, Nothing}
    original_y_labels :: Union{AbstractVector, Nothing}

    function TrainingState(
        X::AbstractLogiset,
        y::AbstractVector{<:CLabel},
        w::Union{Nothing, AbstractVector{<:Real}, Symbol},
        original_y::Union{AbstractVector, Nothing} = nothing,
        original_y_labels::Union{AbstractVector, Nothing} = nothing
    )
        @assert w isa AbstractVector || w in [nothing, :rebalance, :default]

        w = if isnothing(w) || w == :default
            default_weights(y) # ones
        elseif w == :rebalance
            balanced_weights(y)
        else
            w
        end

        !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
        if !isnothing(w)
            !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")
        end

        (ninstances(X) == 0) && error("Empty training set")

        if !isnothing(original_y)
            !(ninstances(X) == length(original_y)) && error("Mismatch number of instances between X and original_y! ($(ninstances(X)) != $(length(original_y)))")
        end

        if !isnothing(original_y_labels)
            !(ninstances(X) == length(original_y_labels)) && error("Mismatch number of instances between X and original_y_labels! ($(ninstances(X)) != $(length(original_y_labels)))")
        end

        return new(X, y, w, original_y, original_y_labels)
    end
end


function sliceinstances(ts::TrainingState, inds::AbstractVector{<:Integer}; return_view=true)
    tr_y = (return_view ? @view(ts.y[inds]) : ts.y[inds] )
    tr_w = (return_view ? @view(ts.w[inds]) : ts.w[inds] )

    tr_original_y = nothing
    if !isnothing(ts.original_y)
        tr_original_y = (return_view) ? @view(ts.original_y[inds]) : ts.original_y[inds]
    end

    tr_original_y_labels = nothing
    if !isnothing(ts.original_y_labels)
        tr_original_y_labels = (return_view) ? @view(ts.original_y_labels[inds]) : ts.original_y_labels[inds]
    end

    return TrainingState(
        slicedataset(ts.X, inds; return_view=return_view),
        tr_y,
        tr_w,
        tr_original_y,
        tr_original_y_labels
    )
end







struct DataSplit
    X :: AbstractLogiset
    y :: AbstractVector{<:CLabel}
    w :: Union{Nothing,AbstractVector{<:Real}}

    grow_inds::Vector{Int}
    prune_inds::Vector{Int}
    permutation_indices::Vector{Int}
end

growth_X(ds::DataSplit; return_view = true) = slicedataset(ds.X, ds.grow_inds; return_view = return_view)
growth_y(ds::DataSplit; return_view = true) = return_view ? @view(ds.y[ds.grow_inds]) : ds.y[ds.grow_inds]

function growth_w(ds::DataSplit; return_view = true)
    isnothing(ds.w) && return nothing 

    return_view ? @view(ds.w[ds.grow_inds]) : ds.w[ds.grow_inds]
end

function prune_X(ds::DataSplit; return_view = true)
    (isnothing(ds.prune_inds) || isempty(ds.prune_inds)) && return []

    slicedataset(ds.X, ds.prune_inds; return_view = return_view)
end

function prune_y(ds::DataSplit; return_view = true)
    (isnothing(ds.prune_inds) || isempty(ds.prune_inds)) && return []

    return_view ? @view(ds.y[ds.prune_inds]) : ds.y[ds.prune_inds]
end

function prune_w(ds::DataSplit; return_view = true)
    isnothing(ds.w) && return nothing 

    return_view ? @view(ds.w[ds.prune_inds]) : ds.w[ds.prune_inds]
end

grow_indices(ds::DataSplit) = ds.grow_inds
prune_indices(ds::DataSplit) = ds.prune_inds
grow_size(ds::DataSplit) = length(ds.grow_inds)
prune_size(ds::DataSplit) = length(ds.prune_inds)
permutation_indices(ds::DataSplit) = ds.permutation_indices



"""
    split_instances(X, y, w, split_ratio, rng; stratified=true, poslabel=1)

Split a labeled dataset into growth and pruning subsets.

This helper function is used during rule learning to reserve a portion of the data for
rule growth while using the remainder for pruning / validation. The returned
`DataSplit` contains indices for the growth set, the pruning set, and the full
permutation used to construct them.

Arguments:
- `X::AbstractLogiset`: input dataset.
- `y::AbstractVector{<:CLabel}`: class labels for each instance.
- `w::AbstractVector{<:Real}`: instance weights.
- `split_ratio::Real`: fraction of the dataset assigned to the growth set.
- `rng::AbstractRNG`: random number generator used to shuffle indices.
- `stratified::Bool`: if `true`, preserve the class distribution of `poslabel`
  between growth and prune subsets.
- `poslabel`: label value treated as the positive class for stratified splitting.

Returns:
- `DataSplit`: object containing `grow_inds`, `prune_inds`, and
  `permutation_indices`.
- `nothing`: if `split_ratio` is too small to assign any instances to the
growth set.
"""
function split_instances(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector{<:Real},
    split_ratio::Real,
    rng::AbstractRNG = Random.default_rng();
    stratified::Bool = true,
    poslabel = 1
)
    n = ninstances(X)
    ngrow_total = round(Integer, n * split_ratio)
    
    # return nothing if the split would put all the data in the prune category
    if ngrow_total == 0
        return nothing
    end

    # handle non-stratified case immediately
    if !stratified
        perm_indices = randperm(rng, n)
        grow_indices = perm_indices[1:ngrow_total]
        prun_indices = perm_indices[ngrow_total+1:end]
        return DataSplit(X, y, w, grow_indices, prun_indices, perm_indices)
    end

    # the case where stratified = true and split_ratio = 1 is handled manually to avoid shuffling and calling findall() for no reason
    if split_ratio == 1.0
        perm_indices = randperm(rng, n)
        return DataSplit(X, y, w, perm_indices, Int[], perm_indices) 
    end

    pos_indices = findall(==(poslabel), y)
    neg_indices = findall(!=(poslabel), y) 
    
    n_pos = length(pos_indices)
    
    # number of positives and negatives that need to go in the grow set
    n_grow_pos = round(Integer, n_pos * split_ratio)
    n_grow_neg = ngrow_total - n_grow_pos
    
    # select the positives and negative groups and shuffle them for random selection between the growth and pruning set
    shuffled_pos = shuffle(rng, pos_indices)
    shuffled_neg = shuffle(rng, neg_indices)
    
    # create the actual grow and prune sets
    grow_indices = vcat(
        shuffled_pos[1:n_grow_pos],
        shuffled_neg[1:n_grow_neg]
    )
    
    prun_indices = vcat(
        shuffled_pos[n_grow_pos+1:end],
        shuffled_neg[n_grow_neg+1:end]
    )

    # shuffle the vectors again as to not have all the positives at first and all the negatives in the end
    shuffle!(rng, grow_indices)
    shuffle!(rng, prun_indices)
    
    perm_indices = vcat(grow_indices, prun_indices)

    return DataSplit(X, y, w, grow_indices, prun_indices, perm_indices)
end



############################################################################################
############ SearchMethods #################################################################
############################################################################################

"""

        SearchMethod

Abstract type for all search methods to be used in [`sequentialcovering`](@ref).
Any search method implements a [`findbestantecedent`](@ref) method.
See also [`findbestantecedent`](@ref), [`BeamSearch`](@ref), [`RandSearch`](@ref).
"""
abstract type SearchMethod end

"""
    findbestantecedent(
        sm::SearchMethod,
        X::AbstractLogiset,
        y::AbstractVector{<:CLabel},
        w::AbstractVector;
        kwargs...
    )

Find the best antecedent formula using `sm` on dataset `X` labelled by `y` and weighted by `w`.
See also [`findbestantecedent`](@ref), [`SearchMethod`](@ref).
"""
function findbestantecedent(
    sm::SearchMethod,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)
    return error("Please, provide method findbestantecedent(sm::$(typeof(sm)), X::$(typeof(X))," *
                 " y::$(typeof(y)), w::$(typeof(w)); kwargs...).")
end

############################################################################################
############ AbstractGenerator #############################################################
############################################################################################
#
"""

        AbstractGenerator

Abstract type representing a generic generator of logical conjuncts.

Subtypes of `AbstractGenerator` are responsible for producing individual conjuncts 
— either atomic predicates or composite formulas — according to specific generation 
rules or constraints. These conjuncts can then be combined using logical `AND` 
operators to form full rule bodies.

Typical use cases include:
- Generating candidate atoms for rule induction.
- Sampling logical subformulas under syntactic or semantic constraints.
- Building the conjunction part of a logical rule (the rule's body).

Implementations should define at least:
- `generate_conjuncts(gen::YourGenerator)`: returns an iterable or vector
  of conjuncts produced by the generator.
"""

abstract type AbstractGenerator end



############################################################################################
############ Feature Selection #############################################################
############################################################################################

"""
Abstract type representing a selector of a certain set S ⊆ F of features selected from
the set of all features F.
"""
abstract type FeatureSelector end


"""
Obtain the set of feature indices to be extracted
"""
function selectfeatures!(fs::FeatureSelector, cols::AbstractVector{Symbol}, rng::AbstractRNG)::Vector{Symbol}
    return error("Please, provide method selectfeatures(fs::$(typeof(fs))")
end

"""
Given a set of selected features, this function extracts from "conditions" the set of conditions that involve
one of the features in selected_features
"""
function extract_conditions(
    conditions::Union{Nothing, Vector{Tuple{Atom, BitVector}}}, 
    selected_features::Vector{Symbol},
    features::Vector{Symbol}
)::Vector{Tuple{Atom, SatMask}}
    
    isnothing(conditions) && return nothing

    # if all the features have been selected, simply return the original list. This can speed up some time when using the default strategy
    if selected_features == features
        return conditions end

    # a Set() allows for O(1) search
    features_set = Set(selected_features)

    filtered_conditions = filter(conditions) do (atom, mask)
        # return atom.value.metacond.feature.i_variable ∈ features_set
        scalar_condition = SoleLogics.value(atom)
        feat = SoleData.feature(scalar_condition)
        return feat.i_variable ∈ features_set
    end

    return Vector{Tuple{Atom, SatMask}}(filtered_conditions)
end