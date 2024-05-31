using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase
using ModalDecisionLists: Measures
using ModalDecisionLists.Measures: laplace_accuracy
using ModalDecisionLists.Measures: significance_test

const SatMask = BitVector



############################################################################################
############ Helping function ##############################################################
############################################################################################

pp(str) = printstyled("$(str) \n", color = :red, bold = true)

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

############################################################################################
############ SearchMethods #################################################################
############################################################################################

"""
Abstract type for all search methods to be used in [`sequentialcovering`](@ref).

findbestantecedent

See also [`BeamSearch`](@ref, [`RandSearch`](@ref).
"""
############################################################################################

abstract type SearchMethod end

"""
For each new SearchMethod, findbestantecedent must be implemented.
"""
function findbestantecedent(
    sm::SearchMethod,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)
    return error("Please, provide method findbestantecedent(sm::$(typeof(sm)), X::$(typeof(X))," *
    " y::$(typeof(y)), w::$(typeof(w)); kwargs...).")
end

############################################################################################
############ AtomSearch ####################################################################
############################################################################################

struct AtomSearch <: SearchMethod end

function findbestantecedent(
    as::AtomSearch,
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector;
    kwargs...
)
    return findbestantecedent(BeamSearch(; conjuncts_search_method=as, max_rule_length=1), X, y, w; kwargs...)
end

############################################################################################


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
    best_satmasks(
        satmasks::Vector{Tuple{Formula, SatMask}},
        y::AbstractVector{CLabel},
        w::AbstractVector,
        beam_width::Integer,
        loss_function::Function
    )

Sort rule satmasks based on their quality, using a specified evaluation function.

Sorts rule antecedents based on their lossfnctn using a specified evaluation function.

Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *lossfnctn evaluator* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the lossfnctn
value of the best one.

See also
[`entropy`](@ref).
"""
function sortantecedents(
    antecedents::AbstractVector{<:Tuple{Formula, BitVector}},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    loss_function::Function,
    min_rule_coverage::Integer,
    maxpurity_gamma::Union{Real, Nothing},
    significance_alpha::Union{Real, Nothing};
    kwargs...
)#= ::Tuple{Vector{Int},<:Real} =#


    # Exit point [1]
    # printstyled(" TARGET[$(counts(y, kwargs[:n_labels])[kwargs[:target_class]])] \n", color=:red )

    isempty(antecedents) && return [], Inf

    if min_rule_coverage > 1
        validindexes = [(count(ant[2]) >= min_rule_coverage) for ant in antecedents
            ] |> findall
        isempty(validindexes) && return [], Inf
        antecedents = antecedents[validindexes]
    end
    indexes = collect(1:length(antecedents))

    antslossfnctn = map(antd -> begin
            _, satinds = antd
            loss_function(y[satinds], w[satinds]; kwargs...)
        end, antecedents)
    if !isnothing(maxpurity_gamma)

        @assert (maxpurity_gamma >= 0) & (maxpurity_gamma <= 1) "maxpurity_gamma not in range [0,1]"

        maxpurity_value = maxpurity_gamma * loss_function(y, w; kwargs...)

        indexes = map(aq -> begin
                        (index, lossfnctn) = aq
                        lossfnctn >= maxpurity_value && index
            end, enumerate(antslossfnctn)
        ) |> filter(x -> x != false)
        isempty(indexes) && return [], Inf
    end
    valid_indexes = partialsortperm(antslossfnctn[indexes], 1:min(beam_width, length(indexes)))

    newstar_perm = indexes[valid_indexes]
    newstar = antecedents[newstar_perm]
    bestantecedent_lossfnctn = antslossfnctn[newstar_perm[1]]

    return newstar, bestantecedent_lossfnctn
end

############################################################################################
############ Utils #########################################################################
############################################################################################
