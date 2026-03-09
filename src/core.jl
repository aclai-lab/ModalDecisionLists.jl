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
