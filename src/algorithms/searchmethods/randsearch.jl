using Parameters
using SoleLogics
using FillArrays
using Random
using ModalDecisionLists.Measures: entropy, significance_test
############################################################################################
############## Random search ###############################################################
############################################################################################

"""
Generate random formulas (`SoleLogics.randformula`)
....
"""
@with_kw mutable struct RandSearch <: SearchMethod
    cardinality::Integer=10
    loss_function::Function=ModalDecisionLists.Measures.entropy
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
    discretizedomain::Bool=false
    rng::Union{Integer,AbstractRNG} = Random.GLOBAL_RNG
    alpha::Real=1.0
    max_purity_const::Union{Real,Nothing}=nothing
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing
end



#
# Potrei mettere in OR regole successive nella DL aventi stessa classe?
function unaryconditions(
    rs::RandSearch,
    a::AbstractAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Formula,SatMask}}

    @unpack cardinality, operators, syntaxheight, rng = rs

    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    # TODO devo generare 10 formule comprese quelle non buone ?
    conditions = [ begin
        formula = randformula(rng, syntaxheight, a, operators)
        satmask = check(formula, X)
        if any(satmask)
            (formula, satmask)
        end
    end for _ in 1:cardinality] |> filter(rf -> rf != nothing)
    #
    return conditions
end


function newconditions(
    rs::RandSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    antecedent::Tuple{Formula,BitVector};
    discretizedomain=false,
    alph::Union{Nothing,AbstractAlphabet}=nothing,
    kwargs...
)::Vector{Tuple{Formula,BitVector}}

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
            alphabet(coveredX;
                discretizedomain = discretizedomain,
                y                = coveredy
            )
        end
    end
    selectedalphabet = UnionAlphabet([ a for a in alphabets(selectedalphabet)])

    # Where new unary conditions are randomly generated (and checked on X) formulas.
    return unaryconditions(rs, selectedalphabet, X)

end

# TODO non mi piaceeee
function extract_optimalantecedent(
    formulas::AbstractVector,
    loss_function,
    max_purity,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    min_rule_coverage::Integer=1,
    kwargs...
)::Tuple{Formula,SatMask}

    # TODO mappare laplace accuracy in intervallo 0,1
    bestant_satmask = ones(Bool, length(y))
    bestformula = begin

        if !isempty(formulas)
            evaluations = map(((rfa, satmask),) -> begin
                relative_loss = loss_function(y[satmask], w[satmask]; kwargs...) - max_purity
                    # TODO @edo review check on min_rule_coverage and min_purity
                    if (relative_loss >= 0) & (count(satmask) > min_rule_coverage)
                        relative_loss
                    else Inf
                    end
            end, formulas)
            bestindex = argmin(evaluations)

            (bestant_formula, bestant_satmask) = formulas[bestindex]
        end
        if all(bestant_satmask) | (evaluations[bestindex] > loss_function(y, w; kwargs...))
            bestant_formula = TOP
            bestant_satmask = ones(length(y))
        end
        (bestant_formula, bestant_satmask)
    end # bestantformula
    return bestformula
end




function findbestantecedent(
    rs::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    min_rule_coverage::Integer,
    kwargs...
)::Tuple{Formula,SatMask}

    @unpack cardinality, loss_function, discretizedomain, default_alphabet,
            operators, syntaxheight, rng, alpha, max_purity_const = rs
    @assert cardinality > 0 "parameter `cardinality` must be greater than zero," * "
                            $(cardinality) is not an acceptable value."
    @assert syntaxheight >= 0 "parameter `syntaxheight` must be greater than zero," * "
                            $(syntaxheight) is not an acceptable value."
    @assert all(o->o isa NamedConnective, operators) "all elements in `operators`" *
                            " must  beNamedConnective"
    max_purity = 0.0
    if !isnothing(max_purity_const)
        @assert (max_purity_const >= 0) & (max_purity_const <= 1) "maxpurity_gamma not in range [0,1]"
        max_purity = loss_function(y, w; kwargs...) * max_purity_const
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
