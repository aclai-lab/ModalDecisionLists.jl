using MLJ: load_iris
using Test
using Random
using SoleBase
using DataFrames
using CategoricalArrays

using SoleBase: CLabel
using ModalDecisionLists
import ModalDecisionLists: maptointeger
import ModalDecisionLists.Measures: entropy, laplace_accuracy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# @with_kw struct RandSearch <: SearchMethod
#     cardinality::Integer=10
#     quality_evaluator::Function=ModalDecisionLists.Measures.entropy
#     operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
#     syntaxheight::Integer=2
# end

# function findbestantecedent(
#     rs::RandSearch,
#     X::PropositionalLogiset,
#     y::AbstractVector{<:CLabel},
#     w::AbstractVector;
#     kwargs...
# )::Tuple{Formula,SatMask}
#     ...
#     return bestantecedent
# end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X...,y = load_iris()

X = PropositionalLogiset(DataFrame(X))

y_clabel = Vector{CLabel}(y)
y_string = Vector{String}(y)
y_catgcl = CategoricalArray(y)

(y_intger, _) = maptointeger(y)

n_instances = ninstances(X)
w = rand(Float16, n_instances)

# Defaul parameters
sequentialcovering(X, y, nrg = MarsenneTwis searchmethod=RandSearch())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cardinality ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# rs = RandSearch(cardinality = -1)
# @test_nowarn sequentialcovering(X, y, searchmethod=rs)
# rs = RandSearch(cardinality = 0)
# @test_nowarn sequentialcovering(X, y, searchmethod=rs)
# rs = RandSearch(cardinality = 1)
# @test_nowarn sequentialcovering(X, y, searchmethod=rs)
