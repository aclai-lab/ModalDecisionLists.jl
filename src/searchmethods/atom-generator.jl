
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleData
using SoleLogics




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
    function checkedatoms_incremental(X::AbstractLogiset, univalph)::Vector{Tuple{Atom,SatMask}}

Like `checkedatoms`, but exploits the monotonic ordering of scalar conditions within a
UnivariateScalarAlphabet to avoid redundant checks.

For `<`  operators the thresholds are sorted descending (most specific first), so each
successive condition can only be satisfied by a *subset* of the previous mask.
For `≥` operators the thresholds are sorted ascending  (most specific first), same logic.

In both cases, once we have mask_prev we only re-check the samples that are still
*active* (i.e. set in mask_prev), because all others are guaranteed to fail.
"""
# function checkedatoms_incremental(X::AbstractLogiset, univalph)::Vector{Tuple{Atom,SatMask}}
#     _atoms = atoms(univalph)
#     result = Tuple{Atom,SatMask}[]
    
#     if isempty(_atoms)
#         return result
#     end

#     # inizalize this with the first atom
#     a_prev = first(_atoms)
#     prev_mask = check(a_prev, X)
#     push!(result, (a_prev, copy(prev_mask)))

#     for a in Iterators.drop(_atoms, 1)
#         # check if we are in the same logical block
#         same_operator = test_operator(a.value) == test_operator(a_prev.value)
#         same_feature  = (a.value.metacond.feature == a.value.metacond.feature)

#         if same_operator && same_feature
#             # we are in the same block: the mask is shrinking
#             active_indices = findall(prev_mask)
#             for i in active_indices
#                 if !check(a, X, i)
#                     prev_mask[i] = false
#                 end
#             end
#             push!(result, (a, copy(prev_mask)))
#         else
#             # the feature or the operator has changed, and the preceding block is over.
#             # we must reset and recalculate the mask from scratch
#             prev_mask = check(a, X)
#             push!(result, (a, copy(prev_mask)))
#         end
        
#         # update the pointer to the previous atom for the next iteration
#         a_prev = a
#     end

#     return result
# end

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
