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


function findbestantecedent(
    rs::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)::Tuple{Formula,SatMask}

    @unpack cardinality, quality_evaluator, operators, syntaxheight, rng = rs

    @assert cardinality > 0 "parameter `cardinality` must be greater than zero," * "
                                $(cardinality) is not an acceptable value."
    # TODO @Gio
    # Un' alternativa sarebbe
    #
    #           isempty(operators) && syntaxheight = 0
    #
    # Le formule diventano di conseguenza degli atomi dato che nessun operatore può
    # essere utilizzato viene utilizzato
    @assert !isempty(operators) "No `operator` for formula construction was provided."

    bestantecedent = begin
        if !allequal(y)
            randformulas = [
                begin
                    rfa = randformula(rng, syntaxheight, alphabet(X), operators)
                    smk = check(rfa, X)
                    # Se la formula non copre nessuna istanza è inutile
                    if any(smk)
                        (rfa, smk)
                    end
                end for _ in 1:cardinality
            ] |> filter(rf -> rf != nothing) # TODO @Gio brutto ?

            (bestant_formula, bestant_satmask) = argmin(((rfa, satmask),) -> begin
                	quality_evaluator(y[satmask], w[satmask])
                end, randformulas)
			# Cosa fare in questo caso ? quando l' antecedente migliore copre tutte le istanze
			# quindi (non siesce a splittare).
			if all(bestant_satmask)
				(⊤, bestant_satmask)
            else
				# bestantecedent
				(bestant_formula,bestant_satmask)
            end
		#
        else
            (⊤, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end



[begin if x > 4
            x
        end
    end for x in [1,2,3,4,5,6,7]
]
