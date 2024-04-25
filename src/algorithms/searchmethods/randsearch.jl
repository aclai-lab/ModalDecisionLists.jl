using Parameters
using SoleLogics
using FillArrays
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
end
function findbestantecedent(
    rs::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)::Tuple{Formula,SatMask}

    @unpack cardinality ,quality_evaluator ,operators ,syntaxheight = rs

    bestantecedent = begin
        if !allequal(y)
            randformulas = [
                begin
                    rfa = randformula(syntaxheight, alphabet(X), operators)
                    (rfa, check(rfa, X))
                end for _ in 1:cardinality
            ]
            (bestant_formula, bestant_satmask) = argmin(((rfa, satmask),) -> begin
                	quality_evaluator(y[satmask], w[satmask])
                end, randformulas)

			# Cosa fare in questo caso ? quando l' antecedente migliore copre tutte le istanze
			# quindi non siesce a splittare le istanze.
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
