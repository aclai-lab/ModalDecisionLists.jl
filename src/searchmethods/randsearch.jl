using Parameters
using SoleLogics: randatom
using SoleData: thresholds
using FillArrays
using Random
using ModalDecisionLists.LossFunctions: entropy, significance_test
############################################################################################
############## Random search ###############################################################
############################################################################################




# TODO @ Gio atompicking_mode ?
# TODO syntaxheight ?
"""
    RandSearch (`SoleLogics.randformula`)

Search method to be used in [`sequentialcovering`](@ref) that explores the solutions space
employing stochastic sampling strategies.

# Keyword Arguments

* `cardinality::Integer = 3`: is the number of formula generated fo the search of a single rule.
the higher the cardinality, the higher the probability of finding a good antecedent.
* `loss_function::Function = entropy`: is the function that assigns a score to each partial solution.
* `operators::Union{Nothing,Integer} = nothing`: specifies the maximum length allowed for a rule in the search algorithm.
* `syntaxheight::Integer=2`: ?
* `discretizedomain::Bool=false`: if true discretizes continuous variables by identifying optimal cut points
* `rng::AbstractRNG=Random.GLOBAL_RNG`
* `alpha::Real=1.0` Actually not implemented
* `max_infogain_ratio::Real=1.0`: maximum information gain for an antecedent with respect to the uncovered training set. Its value is bounded between 0 and 1.
* `default_alphabet::Union{Nothing,AbstractAlphabet}=nothing`: if set, forces the use of a specific alphabet for generating every antecedents. Otherwise
the alphabet is dinamically generated on uncovered instances
* `atompicking_mode::Symbol=:uniform`: allows to bias the probability distribution of MetaConditions in the generation of formulas ...continue"
* `subalphabets_weights::Union{AbstractWeights,AbstractVector{<:Real},Nothing}=nothing`:
"""

@with_kw mutable struct RandSearch <: SearchMethod
    cardinality::Integer=10
    # loss_function::Function=ModalDecisionLists.LossFunctions.entropy
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
    # discretizedomain::Bool=false
    rng::Union{Integer,AbstractRNG} = Random.GLOBAL_RNG
    # significance_alpha::Real=1.0 # Unused
    # max_infogain_ratio::Real=1.0
    # default_alphabet::Union{Nothing,AbstractAlphabet}=nothing
    atompicking_mode::Symbol=:uniform
    subalphabets_weights::Union{AbstractWeights,AbstractVector{<:Real},Nothing} = nothing
end

function unaryconditions(
    rs::RandSearch,
    a::UnionAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Formula,SatMask}}

    @unpack cardinality, operators, syntaxheight, rng,
        atompicking_mode, subalphabets_weights = rs

    conditions = begin
        if natoms(a) == 0
            []
        else
            [ begin
                formula = randformula(rng, syntaxheight, a, operators;
                            atompicker = ( (rng, a) -> SoleLogics.randatom(rng, a;
                                                atompicking_mode = atompicking_mode,
                                                subalphabets_weights = subalphabets_weights
                                ))
                    )
                satmask = check(formula, X)
                if any(satmask)
                    (formula, satmask)
                else nothing
                end
            end for _ in 1:cardinality] |> filter(rf -> rf != nothing)
        end
    end
    #
    return conditions
end

function newconditions(
    rs::RandSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    antecedent::Tuple{Formula,SatMask};
    discretizedomain=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing,
    kwargs...
)::Vector{Tuple{Formula,SatMask}}

    antecedent, satindexes = antecedent
    coveredX = slicedataset(X, satindexes; return_view=false)
    coveredy = y[satindexes]

    # If all instances already belong to the same class,
    # further specializations make no sense.
    allequal(coveredy) && return []

    selectedalphabet = begin
        if !isnothing(alph)
            alph
        else
            alph = alphabet(coveredX;
                discretizedomain = discretizedomain,
                y = coveredy
            )
        end
    end
    # Where new unary conditions are randomly generated (and checked on X) formulas.
    return unaryconditions(rs, selectedalphabet, X)

end

function extract_optimalantecedent(
    formulas::AbstractVector,
    loss_function,
    minloss,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    min_rule_coverage::Integer=1,
    kwargs...
)::Tuple{Formula,SatMask}

    bestant_satmask = ones(Bool, length(y))
    bestformula = begin
        if !isempty(formulas)
            losses = map(((rfa, satmask),) -> begin

                lossfnctn = loss_function(y[satmask], w[satmask]; kwargs...)
                    # TODO @edo review check on min_rule_coverage and min_purity
                    if (lossfnctn >= minloss) && (count(satmask) > min_rule_coverage)
                        lossfnctn
                    else Inf
                    end

            end, formulas)
            bestindex = argmin(losses)
            (bestant_formula, bestant_satmask) = formulas[bestindex]
            if all(bestant_satmask) | (losses[bestindex] > loss_function(y, w; kwargs...))
                bestant_formula = TOP
                bestant_satmask = ones(length(y))
            end
        else
            bestant_formula = TOP
            bestant_satmask = ones(length(y))
        end
        (bestant_formula, bestant_satmask)
    end # bestantformula

    return bestformula
end


# TODO add rng parameter. [UNUSED]
function searchantecedents(
    sm::RandSearch,
    X::PropositionalLogiset,
)::Tuple{Formula,SatMask}

    return (randformula(sm.syntaxheight, alphabet(X), sm.operators) for _ in 1:sm.cardinality)
end

# Called only in RandSearch, not in BeamSearch(;conjuncts_search_method = RandSearch())
function findbestantecedent(
    rs::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    loss_function::Function,
    max_infogain_ratio::Real,
    default_alphabet::Union{Nothing,AbstractAlphabet},
    discretizedomain::Bool,
    significance_alpha::Real,
    min_rule_coverage::Integer;
    #
    kwargs...
)::Tuple{Formula,SatMask}

    @unpack cardinality, operators, syntaxheight, rng = rs

    @assert cardinality > 0 "parameter `cardinality` must be greater than zero," * "
                            $(cardinality) is not an acceptable value."
    @assert syntaxheight >= 0 "parameter `syntaxheight` must be greater than zero," * "
                            $(syntaxheight) is not an acceptable value."
    @assert all(o->o isa NamedConnective, operators) "all elements in `operators`" *
                            " must  beNamedConnective"
    minloss = Inf
    if !isnothing(max_infogain_ratio)
        @assert (max_infogain_ratio >= 0) && (max_infogain_ratio <= 1) "maxpurity_gamma not in range [0,1]"
        minloss = (1-max_infogain_ratio)*loss_function(y, w; kwargs...)
    end
    # isempty(operators) && syntaxheight = 0
    @assert !isempty(operators) "No `operator` for formula construction was provided."
    bestantecedent = begin
        if !allequal(y)
            # select the alphabet parametrization
            selectedalphabet = begin
                if isnothing(default_alphabet)
                    alph = alphabet(X;
                        discretizedomain = discretizedomain,
                        y = y
                    )
                    # If discretizedomain == one it is possible for alph to be empty
                    if discretizedomain && (natoms(alph) == 0)
                        alphabet(X)
                    else alph
                    end
                else
                    default_alphabet
                end
            end
            randformulas = unaryconditions(rs, selectedalphabet, X)
            bestantecedent = extract_optimalantecedent(randformulas,
                            loss_function,
                            minloss,
                            y, w;
                            min_rule_coverage, kwargs...)
        else
            (TOP, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end

# discretizeDomain natoms > 0
#
#
#
