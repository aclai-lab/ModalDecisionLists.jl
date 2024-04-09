using SoleBase: CLabel
using SoleData: PropositionalLogiset
using Parameters
using FillArrays
using StatsBase


const RuleAntecedent = SoleLogics.LeftmostConjunctiveForm{SoleLogics.Atom{ScalarCondition}}
const SatMask = BitVector


############################################################################################
############ SearchMethods #################################################################
############################################################################################

"""
Abstract type for all search methods to be used in [`sequentialcovering`](@ref).

findbestantecedent

See also [`BeamSearch`](@ref, [`RandSearch`](@ref).
"""
abstract type SearchMethod end

"""
For each new SearchMethod, findbestantecedent must be implemented.
"""
function findbestantecedent(
    ::SearchMethod,
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
)
    return error("Please, provide method...")
end

include("algorithms/searchmethods/beamsearch.jl")
include("algorithms/searchmethods/randsearch.jl")

############################################################################################
############ Utils #########################################################################
############################################################################################

function maptointeger(y::AbstractVector{<:CLabel})

    # ordered values
    values = unique(y)
    integer_y = zeros(Int64, length(y))

    for (i, v) in enumerate(values)
        integer_y[y.==v] .= i
    end
    return integer_y, values
end

function entropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end


"""
    best_satmasks(
        satmasks::Vector{Tuple{Formula, SatMask}},
        y::AbstractVector{CLabel},
        w::AbstractVector,
        nbest::Integer,
        quality_evaluator::Function
    )

Sort rule satmasks based on their quality, using a specified evaluation function.

Takes an *satmasks*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *quality evaluator* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the quality
value of the best one.

See also
[`entropy`](@ref).
"""
function best_satmasks(
    satmasks::AbstractVector{BitVector},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    nbest::Integer,
    quality_evaluator::Function,
)::Tuple{Vector{Int},<:Real}
    isempty(satmasks) && return [], Inf

    antsquality = map(satmask->quality_evaluator(y[satmask], w[satmask]), satmasks)

    satmask_perm = partialsortperm(antsquality, 1:min(nbest, length(antsquality)))
    bestantecedent_quality = antsquality[satmask_perm[1]]

    return (satmask_perm, bestantecedent_quality)
end


############################################################################################
############ Helping function ##############################################################
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
