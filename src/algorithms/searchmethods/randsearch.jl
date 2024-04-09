
############################################################################################
############## Random search ###############################################################
############################################################################################

"""
Generate random formulas (`SoleLogics.randformula`)
....
"""
struct RandSearch <: SearchMethod end

# TODO add rng parameter.

function searchantecedents(
    sm::RandSearch,
    X::PropositionalLogiset,
)::Tuple{Formula,SatMask}
    return (randformula(sm.syntaxheight, alphabet(X), sm.operators) for _ in 1:sm.cardinality)
end
function findbestantecedent(
    sm::RandSearch,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    cardinality::Integer=10,
    quality_evaluator::Function=entropy,
    syntaxheight::Integer=2
    operators::AbstractVector=[NEGATION, CONJUNCTION, DISJUNCTION],
)::Tuple{Formula,SatMask}
    bestantecedent = begin
        if !allunique(y)
            randformulas = Iterators.map(φ -> begin
                    satmask = check(rfa, X)
                    (φ, satmask)
                end, searchantecedents(sm, X))
            maximum(((φ, satmask),) -> begin
                    sm.quality_evaluator(y[satmask], w[satmask])
                end, randformulas)
        else
            (⊤, ones(Bool, length(y)))
        end
    end
    return bestantecedent
end
