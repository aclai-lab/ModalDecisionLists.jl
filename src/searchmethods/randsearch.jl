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
* `max_info_gain::Real=1.0`: maximum information gain for an antecedent with respect to the uncovered training set. Its value is bounded between 0 and 1.
* `default_alphabet::Union{Nothing,AbstractAlphabet}=nothing`: if set, forces the use of a specific alphabet for generating every antecedents. Otherwise
the alphabet is dinamically generated on uncovered instances
* `atompicking_mode::Symbol=:uniform`: allows to bias the probability distribution of MetaConditions in the generation of formulas ...continue"
* `subalphabets_weights::Union{AbstractWeights,AbstractVector{<:Real},Nothing}=nothing`:
"""



@with_kw mutable struct RandSearch <: SearchMethod
    cardinality::Integer=10
    loss_function::Function=ModalDecisionLists.LossFunctions.entropy
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
    discretizedomain::Bool=false
    rng::AbstractRNG = Random.GLOBAL_RNG
    alpha::Real=1.0 # Unused
    max_info_gain::Real=1.0
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing
    # randatom parameters
    atompicking_mode::Symbol=:uniform
    subalphabets_weights::Union{AbstractWeights,AbstractVector{<:Real},Nothing} = nothing
end

function unaryconditions(
    rs::RandSearch,
    a::AbstractAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Formula,SatMask}}

    @unpack cardinality, operators, syntaxheight, rng,
        atompicking_mode, subalphabets_weights = rs
    conditions = begin
        if all(isempty, thresholds.(alphabets(a)))
            []
        else
            [ begin
                formula = randformula(rng, syntaxheight, a, operators;
                            atompicker = ( (rng, a) -> randatom(rng, a;
                                                atompicking_mode = atompicking_mode,
                                                subalphabets_weights = subalphabets_weights
                                ))
                    )
                satmask = check(formula, X)
                if any(satmask)
                    (formula, satmask)
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
    max_purity,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    min_rule_coverage::Integer=1,
    kwargs...
)::Tuple{Formula,SatMask}

    bestant_satmask = ones(Bool, length(y))
    bestformula = begin

        if !isempty(formulas)
            losses = map(((rfa, satmask),) -> begin
                relative_loss = loss_function(y[satmask], w[satmask]; kwargs...) - max_purity
                    # TODO @edo review check on min_rule_coverage and min_purity
                    if (relative_loss >= 0) & (count(satmask) > min_rule_coverage)
                        relative_loss
                    else Inf
                    end
            end, formulas)
            bestindex = argmin(losses)

            (bestant_formula, bestant_satmask) = formulas[bestindex]
        end
        if all(bestant_satmask) | (losses[bestindex] > loss_function(y, w; kwargs...))
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
    w::AbstractVector;
    min_rule_coverage::Integer,
    kwargs...
)::Tuple{Formula,SatMask}

    @unpack cardinality, loss_function, discretizedomain, default_alphabet,
            operators, syntaxheight, rng, alpha, max_info_gain = rs
    @assert cardinality > 0 "parameter `cardinality` must be greater than zero," * "
                            $(cardinality) is not an acceptable value."
    @assert syntaxheight >= 0 "parameter `syntaxheight` must be greater than zero," * "
                            $(syntaxheight) is not an acceptable value."
    @assert all(o->o isa NamedConnective, operators) "all elements in `operators`" *
                            " must  beNamedConnective"
    max_purity = 0.0
    if !isnothing(max_info_gain)
        @assert (max_info_gain >= 0) & (max_info_gain <= 1) "maxpurity_gamma not in range [0,1]"
        max_purity = loss_function(y, w; kwargs...) * max_info_gain
    end
    # isempty(operators) && syntaxheight = 0
    @assert !isempty(operators) "No `operator` for formula construction was provided."
    bestantecedent = begin
        if !allequal(y)
            # select the alphabet parametrization
            selectedalphabet = begin
                if isnothing(default_alphabet)
                    alphabet(X;
                        discretizedomain = discretizedomain,
                        y = y
                    )
                else default_alphabet
                end
            end
            randformulas = unaryconditions(rs, selectedalphabet, X)
            bestantecedent = extract_optimalantecedent(randformulas,
                            loss_function, max_purity, y, w;
                            min_rule_coverage, kwargs...)
        else
            (TOP, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end
