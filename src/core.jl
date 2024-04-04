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

function soleentropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end

function sortantecedents(
    antecedents::AbstractVector{<:Tuple{RuleAntecedent, BitVector}},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    quality_evaluator::Function,
)::Tuple{Vector{Int},<:Real}
    isempty(antecedents) && return [], Inf

    antsquality = map(antd -> begin
            _, satinds = antd
            quality_evaluator(y[satinds], w[satinds])
        end, antecedents)

    newstar_perm = partialsortperm(antsquality, 1:min(beam_width, length(antsquality)))
    bestantecedent_quality = antsquality[newstar_perm[1]]

    return (newstar_perm, bestantecedent_quality)
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
