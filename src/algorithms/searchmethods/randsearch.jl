using Parameters
using SoleLogics
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
        if !allunique(y)
            randformulas = [
                begin
                    rfa = randformula(rs.syntaxheight, alphabet(X), rs.operators)
                    (rfa, check(rfa, X))
                end for _ in 1:rs.cardinality
            ]
            argmax(((rfa, satmask),) -> begin
                    rs.quality_evaluator(y[satmask], w[satmask])
                end, randformulas)[1]
        else
            (⊤, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end
