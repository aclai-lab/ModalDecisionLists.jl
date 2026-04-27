using Parameters
using SoleLogics: randatom
using SoleData: thresholds
using FillArrays
using Random
using ModalDecisionLists.Metrics: entropy, significance_test
############################################################################################
############## Random search ###############################################################
############################################################################################

"""
    RandSearch (`SoleLogics.randformula`)

Search method to be used in [`sequentialcovering`](@ref) that explores the solutions space
employing stochastic sampling strategies.

# Keyword Arguments

* `cardinality::Integer=25`: Defines the number of formulas generated during the search for a single rule. A higher cardinality increases the probability of finding an antecedent that better fits the data.
* `operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]`: Represents the set of logical operators used in the generation of formulas.
* `syntaxheight::Integer=2`: Defines the maximum height of the syntactic tree representing a generated formula.
* `rng::AbstractRNG=Random.GLOBAL_RNG`: Specifies the random number generator to be used in the generation of formulas. By default, it uses the global random number generator.
* `atompicking_mode::Symbol=:uniform`: Determines the probability distribution of MetaConditions when generating formulas. It can impose a :uniform distribution over the MetaConditions or a :weighted distribution based on the length of the thresholding values of each MetaCondition.
* `subalphabets_weights::Union{AbstractWeights,AbstractVector{<:Real},Nothing}=nothing`: Allows biasing the probability distribution of each MetaCondition through a vector of real weights between 0 and 1.

See also
[`sequentialcovering`](@ref),
[`SearchMethod`](@ref),
[`BeamSearch`](@ref),
[`specializeantecedents`](@ref).
"""
@kwdef mutable struct RandomGenerator <: AbstractGenerator
    cardinality::Integer=10
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
    rng::Union{Integer,AbstractRNG} = Random.GLOBAL_RNG
    atompicking_mode::Symbol=:uniform
    subalphabets_weights::Union{AbstractWeights,AbstractVector{<:Real},Nothing} = nothing
end

"""
    alphabet2conditions(gen::RandomGenerator, alphabet::UnionAlphabet, X::AbstractLogiset)
        -> Vector{Tuple{Formula, SatMask}}

Generates a collection of logical conditions (formula + satisfiability mask)
from a given logical alphabet and a dataset `X`.

# Arguments
- `gen::RandomGenerator`: an instance containing generation parameters such as
  `cardinality`, `operators`, `syntaxheight`, `rng`, `atompicking_mode`,
  and `subalphabets_weights`.
- `alphabet::UnionAlphabet`: the logical alphabet from which atoms are sampled.
- `X::AbstractLogiset`: the logical dataset on which formulas are evaluated.

# Returns
A vector of tuples `(formula, satmask)` where:
- `formula::Formula` is a randomly generated logical formula.
- `satmask::SatMask` is a Boolean mask indicating where the formula
  is satisfied over `X`.

# Notes
Only formulas that are satisfied in at least one instance of `X`
(i.e. `any(satmask) == true`) are kept in the final result.
"""
function alphabet2conditions(
    gen::RandomGenerator,
    alphabet::UnionAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Formula, SatMask}}

    # Unpack main generation parameters
    @unpack cardinality, operators, syntaxheight, rng,
        atompicking_mode, subalphabets_weights = gen

    # Early exit if the alphabet contains no atoms
    natoms(alphabet) == 0 && return Tuple{Formula, SatMask}[]
    
    atompicker = (rng, a) -> SoleLogics.randatom(rng, a;
        atompicking_mode,
        subalphabets_weights,
    )
 
    # Generate candidate formulas and keep only satisfiable ones
    conditions = Tuple{Formula, SatMask}[]
    for _ in 1:cardinality
        formula = randformula(rng, syntaxheight, alphabet, operators; atompicker)
        satmask = check(formula, X)
        if any(satmask)  # keep only formulas satisfied somewhere in X
            push!(conditions, (formula, satmask))
        end
    end

    return conditions
end


# function newconditions(
#     rs::RandSearch,
#     X::AbstractLogiset,
#     y::AbstractVector{<:CLabel},
#     antecedent::Antecedent,

#     discretizedomain=false,
#     default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
# )::Vector{Tuple{Formula,SatMask}}

#     antecedent, satindexes = antecedent
#     _X = slicedataset(X, satindexes; return_view=false)
#     _y = y[satindexes]

#     # If all instances already belong to the same class,
#     # further specializations make no sense.
#     allequal(coveredy) && return []

#     selectedalphabet = begin
#         if !isnothing(default_alphabet)
#             default_alphabet
#         else
#             default_alphabet = alphabet(coveredX;
#                 discretizedomain = discretizedomain,
#                 y = coveredy
#             )
#         end
#     end
#     # Where new unary conditions are randomly generated (and checked on X) formulas.
#     return unaryconditions(rs, selectedalphabet, X)

# end

# function extract_optimalantecedent(
#     formulas::AbstractVector,
#     loss_function,
#     minloss,
#     y::AbstractVector{<:CLabel},
#     w::AbstractVector;
#     min_rule_coverage::Integer=1,
#     kwargs...
# )::Tuple{Formula,SatMask}

#     bestant_satmask = ones(Bool, length(y))
#     bestformula = begin
#         if !isempty(formulas)
#             losses = map(((rfa, satmask),) -> begin

#                 lossfnctn = loss_function(y[satmask], w[satmask]; kwargs...)
#                     # TODO @edo review check on min_rule_coverage and min_purity
#                     if (lossfnctn >= minloss) && (count(satmask) > min_rule_coverage)
#                         lossfnctn
#                     else Inf
#                     end

#             end, formulas)
#             bestindex = argmin(losses)
#             (bestant_formula, bestant_satmask) = formulas[bestindex]
#             if all(bestant_satmask) | (losses[bestindex] > loss_function(y, w; kwargs...))
#                 bestant_formula = TOP
#                 bestant_satmask = ones(length(y))
#             end
#         else
#             bestant_formula = TOP
#             bestant_satmask = ones(length(y))
#         end
#         (bestant_formula, bestant_satmask)
#     end # bestantformula

#     return bestformula
# end


# # TODO add rng parameter. [UNUSED]
# function searchantecedents(
#     sm::RandSearch,
#     X::PropositionalLogiset,
# )::Tuple{Formula,SatMask}

#     return (randformula(sm.syntaxheight, alphabet(X), sm.operators) for _ in 1:sm.cardinality)
# end

# # Called only in RandSearch, not in BeamSearch(;conjuncts_search_method = RandSearch())
# function findbestantecedent(
#     rs::RandSearch,
#     X::PropositionalLogiset,
#     y::AbstractVector{<:CLabel},
#     w::AbstractVector,
#     loss_function::Function,
#     max_infogain_ratio::Real,
#     default_alphabet::Union{Nothing,AbstractAlphabet},
#     discretizedomain::Bool,
#     significance_alpha::Real,
#     min_rule_coverage::Integer;
#     #
#     kwargs...
# )::Tuple{Formula,SatMask}

#     @unpack cardinality, operators, syntaxheight, rng = rs

#     @assert cardinality > 0 "parameter `cardinality` must be greater than zero," * "
#                             $(cardinality) is not an acceptable value."
#     @assert syntaxheight >= 0 "parameter `syntaxheight` must be greater than zero," * "
#                             $(syntaxheight) is not an acceptable value."
#     @assert all(o->o isa NamedConnective, operators) "all elements in `operators`" *
#                             " must  beNamedConnective"
#     minloss = Inf
#     if !isnothing(max_infogain_ratio)
#         @assert (max_infogain_ratio >= 0) && (max_infogain_ratio <= 1) "maxpurity_gamma not in range [0,1]"
#         minloss = (1-max_infogain_ratio)*loss_function(y, w; kwargs...)
#     end
#     # isempty(operators) && syntaxheight = 0
#     @assert !isempty(operators) "No `operator` for formula construction was provided."
#     bestantecedent = begin
#         if !allequal(y)
#             # select the alphabet parametrization
#             selectedalphabet = begin
#                 if isnothing(default_alphabet)
#                     alph = alphabet(X;
#                         discretizedomain = discretizedomain,
#                         y = y
#                     )
#                     # If discretizedomain == one it is possible for alph to be empty
#                     if discretizedomain && (natoms(alph) == 0)
#                         alphabet(X)
#                     else
#                         alph
#                     end
#                 else
#                     default_alphabet
#                 end
#             end
#             randformulas = unaryconditions(rs, selectedalphabet, X)
#             bestantecedent = extract_optimalantecedent(randformulas,
#                             loss_function,
#                             minloss,
#                             y, w;
#                             min_rule_coverage, kwargs...)
#         else
#             (TOP, ones(Bool, length(y)))
#         end
#     end
#     return bestantecedent
# end

# discretizeDomain natoms > 0
#
#
#
