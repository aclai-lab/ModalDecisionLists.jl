
using SoleData: AbstractLogiset, PropositionalLogiset




############################################################################################
############ AtomSearch ####################################################################
############################################################################################
"""

        AtomGenerator

Method for conjunctions search to be used with [`BeamSearch`](@ref), where the conjunctions are restricted to
atomic conditions. This approach precisely implements the CN2 algorithm.
"""
struct AtomGenerator <: AbstractGenerator end

"""
    function checkedatoms(X::AbstractLogiset, alph)::Vector{Tuple{Atom,SatMask}}

Given a set of samples/interpretations X and an alphabet alph, it returns a list of tuples
(a, mask) where each atom is mapped to its corresponding SatMask on X.
"""
checkedatoms(X::AbstractLogiset, alph)::Vector{Tuple{Atom,SatMask}} = [(a, check(a, X)) for a ∈ atoms(alph)]

"""
    function checkedatoms( TODO )

..... TODO
"""
function alphabet2conditions(
    ::AtomGenerator,
    a::UnionAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Atom,SatMask}}

    _conditions = Tuple{Atom{ScalarCondition},SatMask}[]

    for univalph in subalphabets(a)
        newconds = checkedatoms(X, univalph)
        append!(_conditions, newconds)
    end
    return _conditions
end
