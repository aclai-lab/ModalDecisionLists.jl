
############################################################################################
############## Random search ###############################################################
############################################################################################

"""
Generate random formulas (`SoleLogics.randformula`)
....
"""
@with_kw struct RandSearch <: SearchMethod
    cardinality::Integer=10
    quality_evaluator::Function=soleentropy
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION]
    syntaxheight::Integer=2
end

function findbestantecedent(
    sm::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector
)::Tuple{Formula,SatMask}
    bestantecedent = begin
        if !allunique(y)
            randformulas = [
                begin
                    rfa = randformula(sm.syntaxheight, alphabet(X), sm.operators)
                    (rfa, check(rfa, X))
                end for _ in 1:sm.cardinality
            ]
            argmax(((rfa, satmask),) -> begin
                    sm.quality_evaluator(y[satmask], w[satmask])
                end, randformulas)[1]
        else
            (⊤, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end
