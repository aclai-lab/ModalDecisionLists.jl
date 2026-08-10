
using SoleData: AbstractLogiset, PropositionalLogiset, UnivariateScalarAlphabet
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
    X::AbstractLogiset,
    discretizedomain::Bool = false
)::Vector{Tuple{Atom,SatMask}}

    _conditions = Tuple{Atom{ScalarCondition},SatMask}[]

    for subalph in subalphabets(a)
        # newconds = checkedatoms(X, subalph)
        # append!(_conditions, newconds)

        # if the domain is discretized, it makes no sense to create tests only in between two successive feature values, because
        # those values are already the points at which the splits make the most sense
        if discretizedomain
            newconds = checkedatoms(X, subalph)
            append!(_conditions, newconds)
            continue
        end

        if subalph isa UnionAlphabet
            append!(_conditions, alphabet2conditions(AtomGenerator(), subalph, X))
        elseif subalph isa UnivariateScalarAlphabet
            threshs = SoleData.thresholds(subalph)
            if !isempty(threshs) && eltype(threshs) <: Number
                append!(_conditions, midpointconditions(X, subalph))
            else
                append!(_conditions, checkedatoms(X, subalph))
            end
        else
            append!(_conditions, checkedatoms(X, subalph))
        end
    end
    return _conditions
end

"""
    midpointconditions(X::AbstractLogiset, univalph::UnivariateScalarAlphabet)

Generate additional scalar conditions at the midpoints between consecutive threshold
values of a numeric feature. This is useful for creating split points that lie
between observed values, e.g. turning a threshold pair `[3, 4]` into a midpoint at
`3.5` for conditions such as `< 3.5` or `≥ 3.5`.
"""
function midpointconditions(
    X::AbstractLogiset,
    univalph::UnivariateScalarAlphabet
)::Vector{Tuple{Atom,SatMask}}

    thresholds = collect(SoleData.thresholds(univalph))
    thresholds = sort(unique(thresholds))
    isempty(thresholds) && return Tuple{Atom,SatMask}[]

    mc = metacond(univalph)
    feature = SoleData.feature(mc)

    conditions = Tuple{Atom{ScalarCondition},SatMask}[]
    
    for i in 1:(length(thresholds) - 1)
        prev_threshold = thresholds[i]
        next_threshold = thresholds[i + 1]

        midpoint = (prev_threshold + next_threshold) / 2
        if midpoint == prev_threshold || midpoint == next_threshold
            continue
        end

        new_mc_lt = ScalarMetaCondition(feature, <)
        new_mc_ge = ScalarMetaCondition(feature, ≥)

        atom_lt = Atom(ScalarCondition(new_mc_lt, midpoint))
        atom_ge = Atom(ScalarCondition(new_mc_ge, midpoint))
        
        push!(conditions, (atom_lt, check(atom_lt, X)))
        push!(conditions, (atom_ge, check(atom_ge, X)))
    end

    # manually insert the upper threshold 
    max_threshold = thresholds[end]

    atom_lt = Atom(ScalarCondition(ScalarMetaCondition(feature, <), max_threshold))
    atom_ge = Atom(ScalarCondition(ScalarMetaCondition(feature, ≥), max_threshold))

    push!(conditions, (atom_lt, check(atom_lt, X)))
    push!(conditions, (atom_ge, check(atom_ge, X)))

    return conditions
end
