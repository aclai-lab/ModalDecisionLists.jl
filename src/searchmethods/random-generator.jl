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


