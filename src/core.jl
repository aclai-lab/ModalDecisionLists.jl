using DataFrames

using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase
using ModalDecisionLists.LossFunctions: laplace_accuracy
using ModalDecisionLists.LossFunctions: significance_test

const SatMask = BitVector


############################################################################################
############ Utilities #####################################################################
############################################################################################


#=

using RDatasets
using SoleBase: CLabel
using Revise
using ModalDecisionLists
iris = dataset("datasets", "iris")
X = PropositionalLogiset(iris[:, 1:4])
y = Vector{CLabel}(iris[:, 5])
sequentialcovering(X,y)

LeftmostConjunctiveForm([
Atom( ScalarCondition(ScalarMetaCondition(Feature(:w), <), 3))
Atom( ScalarCondition(ScalarMetaCondition(Feature(:m), <), 3))
Atom( ScalarCondition(ScalarMetaCondition(Feature(:w), <), 3))
Atom( ScalarCondition(ScalarMetaCondition(Feature(:w), <), 3))
])

=#

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
struct InstanceSet
    X::AbstractLogiset
    y::AbstractVector{<:CLabel}
    w::Union{Nothing,AbstractVector{<:Real},Symbol}

    indices::BitVector  # TODO: da cambiare in Satmask
end

# Contruttore
# function InstanceSet(
#     X::AbstractLogiset, 
#     y::AbstractVector{<:CLabel}, 
#     w::Union{Nothing,AbstractVector{<:Real},Symbol}=nothing
# )
#     @assert w isa AbstractVector || w in [nothing, :rebalance, :default]
#
#     w = if isnothing(w) || w == :default
#         default_weights(y) # ones
#     elseif w == :rebalance
#         balanced_weights(y)
#     else
#         w
#     end
#
#     # in Parameters.jl
#
#     !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
#     !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")
#     (ninstances(X) == 0) && error("Empty trainig set")
#
#     return InstanceSet(X, y, w, zeros(Bool, length(y)))
# end
#
# function sliceinstances(inset::InstanceSet, mask::SatMask)
#
#     uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
#     uncoveredy = @view uncoveredy[uncovered_slice]
#     uncoveredw = @view uncoveredw[uncovered_slice]
# end
#




Base.show(io::IO, inst::InstanceSet) = begin
    println(io, "InstanceSet with $(ninstances(inst.X)) instances")
    println(io, inst.X[1, :])
    # Tipo del vettore y
    println(io, "\n  y: ", typeof(inst.y))
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


include("searchmethods/beamsearch.jl")
include("searchmethods/randsearch.jl")
include("searchmethods/atom-generator.jl")
include("searchmethods/random-generator.jl")


function maptointeger(y::AbstractVector{<:CLabel})

    # ordered values
    values = unique(y)
    integer_y = zeros(UInt32, length(y))

    for (i, v) in enumerate(values)
        integer_y[y.==v] .= i
    end
    return integer_y, values
end

"""
    TODO sortantecedents....

Sort rule satmasks based on their loss, using a specified loss function.

Sorts rule antecedents based on their lossfnctn using a specified loss function.

Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *loss_function* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the lossfnctn
value of the best one.

See alsoì com’è, non funziona: c’è ancora un problema di tipo nella prima riga del costruttore.
[`entropy`](@ref).
"""
function sortantecedents(
    antecedents::AbstractVector{Antecedent},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    loss_function::Function,
    min_rule_coverage::Integer,
    max_infogain_ratio::Union{Real,Nothing},
    significance_alpha::Union{Real,Nothing};
    kwargs...
)::Tuple{AbstractVector,<:Real}

    isempty(antecedents) && return [], Inf

    # TODO @Edo Fix !
    #
    # if min_rule_coverage > 1
    #     validindexes = [(count(ant[2]) >= min_rule_coverage) for ant in antecedents
    #     ] |> findall
    #     isempty(validindexes) && return [], Inf
    #     antecedents = antecedents[validindexes]
    # end

    indexes = eachindex(antecedents)

    antslossfnctn = map(a ->  loss_function(y[a.covmask], w[a.covmask]; kwargs...) , antecedents)


    if !isnothing(max_infogain_ratio)
        minloss = (1-max_infogain_ratio)*loss_function(y, w; kwargs...)

        # Filter antecedents (their indexes by minimum loss), equivalente al codice commentato sotto
        indexes = [ind for (ind, loss) in enumerate(antslossfnctn) if loss ≥ minloss]
        # indexes = map(aq -> 
        #     begin
        #         (index, lossfnctn) = aq
        #         (lossfnctn >= minloss) && index
        #     end, 
        #     enumerate(antslossfnctn)
        # ) |> filter(x -> x != false)

        isempty(indexes) && return [], Inf
    end
    valid_indexes = partialsortperm(antslossfnctn[indexes], 1:min(beam_width, length(indexes)))

    newstar_perm = indexes[valid_indexes]
    newstar = antecedents[newstar_perm]
    bestantecedent_lossfnctn = antslossfnctn[newstar_perm[1]]

    return newstar, bestantecedent_lossfnctn
end

############################################################################################
############ Utils #########################################################################
############################################################################################

function preprocess_inputdata(
    X::AbstractDataFrame,
    y;
    remove_duplicate_rows=false
)
    if remove_duplicate_rows
        allunique(X) && return (X, y)
        nonunique_ind = nonunique(X)
        Xy = hcat(X[findall((!).(nonunique_ind)), :],
            y[findall((!).(nonunique_ind))]
        ) |> dropmissing
    else
        Xy = hcat(X[:, :], y[:]) |> dropmissing
    end
    return Xy[:, 1:(end-1)], Xy[:, end]
end
