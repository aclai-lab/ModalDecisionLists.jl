
const RuleAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom{ScalarCondition}}

function checkconditionsequivalence(
    φ1::RuleAntecedent,
    φ2::RuleAntecedent,
)::Bool
    return  length(φ1) == length(φ2) &&
            !any(iszero, map( x-> x ∈ atoms(φ1), atoms(φ2)))
end
