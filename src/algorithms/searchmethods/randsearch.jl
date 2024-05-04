using Parameters
using SoleLogics
using FillArrays
using Random
############################################################################################
############## Random search ###############################################################
############################################################################################

"""
Generate random formulas (`SoleLogics.randformula`)
....
"""
@with_kw struct RandSearch <: SearchMethod
    cardinality::Integer=10
    quality_evaluator::Function=ModalDecisionLists.Measures.entropy
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
    rng::Union{Integer,AbstractRNG} = Random.GLOBAL_RNG
    max_purity_const::Union{Real,Nothing}=nothing
end

# TODO non mi piaceeee
function extract_optimalantecedent(
    formulas::AbstractVector,
    quality_evaluator,
    max_purity,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)::Tuple{Formula,SatMask}
    bestant_satmask = ones(Bool, length(y))
    bestformula = begin
        if !isempty(formulas)
            (bestant_formula, bestant_satmask) = argmin(((rfa, satmask),) -> begin
                relative_quality = quality_evaluator(y[satmask], w[satmask]; kwargs...) - max_purity
                if relative_quality >= 0
                    relative_quality
                else Inf
                end
            end, formulas)
        end
        #
        if all(bestant_satmask)
            bestant_formula = TOP
        end
        (bestant_formula, bestant_satmask)
    end
    return bestformula
end


function findbestantecedent(
    rs::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)::Tuple{Formula,SatMask}

    @unpack cardinality, quality_evaluator,
            operators, syntaxheight, rng, max_purity_const = rs
    @assert cardinality > 0 "parameter `cardinality` must be greater than zero," * "
                            $(cardinality) is not an acceptable value."
    @assert syntaxheight >= 0 "parameter `syntaxheight` must be greater than zero," * "
                            $(syntaxheight) is not an acceptable value."
    @assert all(o->o isa NamedConnective, operators) "all elements in `operators`" *
                            " must  beNamedConnective"
    max_purity = 0.0
    if !isnothing(max_purity_const)
        @assert (max_purity_const >= 0) & (max_purity_const <= 1) "maxpurity_gamma not in range [0,1]"
        max_purity = quality_evaluator(y, w; kwargs...) * max_purity_const
    end
    # isempty(operators) && syntaxheight = 0
    @assert !isempty(operators) "No `operator` for formula construction was provided."
    bestantecedent = begin
        if !allequal(y)
            randformulas = [ begin
                    rfa = randformula(rng, syntaxheight, alphabet(X), operators)
                    smk = check(rfa, X)
                    if any(smk)
                        (rfa, smk)
                    end
                end for _ in 1:cardinality
            ] |> filter(rf -> rf != nothing) # TODO @Gio brutto ?
            bestantecedent = extract_optimalantecedent(randformulas,
                            quality_evaluator, max_purity, y, w;
                            kwargs...)
        else
            (TOP, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end
