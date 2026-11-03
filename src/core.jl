using DataFrames

using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase

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
    split_instances(
        X::AbstractLogiset,
        y::AbstractVector{<:CLabel},
        w::AbstractVector{<:Real},
        split_ratio::Real,
        rng::AbstractRNG = Random.default_rng()
    ) -> Union{DataSplit, Nothing}

Split the dataset `(X, y, w)` into two disjoint subsets: a "grow" set and a "prune" set,
based on the specified `split_ratio`. The split is performed randomly using the provided
random number generator `rng`.

# Arguments
- `X::AbstractLogiset`: The input dataset, containing the instances to be split.
- `y::AbstractVector{<:CLabel}`: The labels corresponding to the instances in `X`.
- `w::AbstractVector{<:Real}`: The weights associated with each instance in `X`.
- `split_ratio::Real`: The proportion of instances to allocate to the "grow" set.
- `rng::AbstractRNG`: The random number generator to use for shuffling the instances.

# Returns
- `DataSplit`: An object containing the split dataset information, including:
  - The original `X`, `y`, and `w`.
  - `grow_inds`: Indices of instances in the grow set.
  - `prune_inds`: Indices of instances in the prune set.
  - `permutation_indices`: The random permutation used for splitting.
- `Nothing`: If the split would result in an empty grow set or an empty prune set
  (i.e., if `round(n * split_ratio) == 0` or `round(n * split_ratio) == n`, where `n`
  is the number of instances).

# Notes
- The split is performed by generating a random permutation of instance indices and
  assigning the first `ngrow = round(Integer, n * split_ratio)` indices to the grow set,
  with the remaining indices going to the prune set.
- The function ensures that both subsets are non-empty to avoid degenerate splits.

See also: [`DataSplit`](@ref), [`growth_X`](@ref), [`prune_X`](@ref).
"""
function split_instances(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector{<:Real},
    split_ratio::Real,
    rng::AbstractRNG = Random.default_rng()
)
    n = ninstances(X)
    ngrow = round(Integer, n * split_ratio)

    # return nothing if the split would put all the data either in the grow category or in the prune category
    if ngrow == 0 # || n - ngrow == 0
        return nothing
    end

    perm_indices = randperm(rng, n)
    grow_indices = perm_indices[1:ngrow]
    prun_indices = perm_indices[ngrow+1:end]

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