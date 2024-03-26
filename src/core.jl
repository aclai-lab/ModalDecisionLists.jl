
<<<<<<< HEAD
const RuleAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom{ScalarCondition}}

function checkconditionsequivalence(
    φ1::RuleAntecedent,
    φ2::RuleAntecedent,
)::Bool
    return  length(φ1) == length(φ2) &&
            !any(iszero, map( x-> x ∈ atoms(φ1), atoms(φ2)))
=======
############################################################################################
############ helping function ##############################################################
############################################################################################

macro showlc(list, c)
    return esc(quote
        infolist = (length($list) == 0 ?
                        "EMPTY" :
                        "len: $(length($list))"
                    )
        printstyled($(string(list)),  " | $infolist \n", bold=true, color=$c)
        for (ind, element) in enumerate($list)
            printstyled(ind,") ",element, "\n", color=$c)
        end
    end)

>>>>>>> edo
end
