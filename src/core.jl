


const RuleAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom{ScalarCondition}}
# const RuleAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom}
const SatMask = BitVector


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

end
