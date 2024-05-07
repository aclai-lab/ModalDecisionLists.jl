using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase
using ModalDecisionLists: Measures
using ModalDecisionLists.Measures: laplace_accuracy

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
    sm::SearchMethod,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwards...
)
    return error("Please, provide method findbestantecedent(sm::$(typeof(sm)), X::$(typeof(X))," *
    " y::$(typeof(y)), w::$(typeof(w)); kwargs...).")
end

include("algorithms/searchmethods/beamsearch.jl")
include("algorithms/searchmethods/randsearch.jl")

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
    y::AbstractVector{<:Integer},
    w::AbstractVector,
    beam_width::Integer,
    quality_evaluator::Function,
    maxpurity_gamma::Union{Real, Nothing}=nothing;
    kwargs...
)::Tuple{Vector{Int},<:Real}


    # Exit point [1]

    isempty(antecedents) && return [], Inf
    indexes = collect(1:length(antecedents))

    antsquality = map(antd -> begin
            _, satinds = antd
            quality_evaluator(y[satinds], w[satinds]; kwargs...)
        end, antecedents)

    for i in 1:length(antsquality)
        println("E: $(counts(y[antecedents[i][2]], 3)) $(antsquality[i])")
    end


    if !isnothing(maxpurity_gamma)
        # TODO va fatto qui questa @assert o in `findbestantecedent` ?
        @assert (maxpurity_gamma >= 0) & (maxpurity_gamma <= 1) "maxpurity_gamma not in range [0,1]"
        maxpurity_value = maxpurity_gamma * quality_evaluator(y, w; kwargs...)
        indexes = map(aq -> begin
                        (index, quality) = aq
                        quality >= maxpurity_value && index
            end, enumerate(antsquality)
        ) |> filter(x -> x != false)
        # Exit point [2]
        isempty(indexes) && return [], Inf
    end
    valid_indexes = partialsortperm(antsquality[indexes], 1:min(beam_width, length(indexes)))
    newstar_perm = indexes[valid_indexes]
    bestantecedent_quality = antsquality[newstar_perm[1]]
    @show bestantecedent_quality

    return newstar_perm, bestantecedent_quality
end

############################################################################################
############ Utils #########################################################################
############################################################################################


"""
Dumb utility function to preprocess input data:
    * remove duplicated rows
    * remove rows with missing values
"""
function preprocess_inputdata(
    X::AbstractDataFrame,
    y;
    remove_duplicate_rows = false
)
    if remove_duplicate_rows
        allunique(X) && return (X, y)
        nonunique_ind = nonunique(X)
        Xy = hcat( X[findall((!).(nonunique_ind)), :],
                   y[findall((!).(nonunique_ind))]
        ) |> dropmissing
    else
        Xy = hcat(X[:, :], y[:]) |> dropmissing
    end
    return Xy[:, 1:(end-1)], Xy[:, end]
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
