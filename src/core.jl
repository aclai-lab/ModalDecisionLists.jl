using SoleBase: CLabel
using SoleData: PropositionalLogiset
using SoleModels: bestguess
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

# TODO è come se diventasse un problema biclasse ?
"""
class LaplaceAccuracyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        # as an exception, when target class is not set,
        # the majority class is chosen to stand against
        # all others
        tc = rule.target_class
        dist = rule.curr_class_dist
        if tc is not None:
            k = 2
            target = dist[tc]
        else:
            k = len(dist)
            target = bn.nanmax(dist)
        return (target + 1) / (dist.sum() + k)
"""

function sole_laplace_estimator(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
    n_labels::Integer = 2
)
    N = length(y)
    target_class = SoleModels.bestguess(y, suppress_parity_warning=true)
    n = countmap(y)[target_class]

    return -(n + 1) / (N + n_labels)
end

function soleentropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
    kwargs...
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    return -sum(prob .* log2.(prob))
end

function maptointeger(y::AbstractVector{<:CLabel})

    # ordered values
    values = unique(y)
    integer_y = zeros(Int64, length(y))

    for (i, v) in enumerate(values)
        integer_y[y.==v] .= i
    end
    return integer_y, values
end

"""
    sortantecedents(
        antecedents::Vector{Tuple{RuleAntecedent, SatMask}},
        y::AbstractVector{CLabel},
        w::AbstractVector,
        beam_width::Integer,
        quality_evaluator::Function
    )


Sorts rule antecedents based on their quality using a specified evaluation function.

Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *quality evaluator* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the quality
value of the best one.
"""
function sortantecedents(
    antecedents::AbstractVector{<:Tuple{RuleAntecedent, BitVector}},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    quality_evaluator::Function;
    kwargs...
)::Tuple{Vector{Int},<:Real}
    isempty(antecedents) && return [], Inf
    antsquality = map(antd -> begin
            _, satinds = antd
            quality_evaluator(y[satinds], w[satinds]; kwargs...)
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
