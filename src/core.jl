using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase
using ModalDecisionLists.LossFunctions: laplace_accuracy
using ModalDecisionLists.LossFunctions: significance_test

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

        SearchMethod

Abstract type for all search methods to be used in [`sequentialcovering`](@ref).

Any search method implements a [`findbestantecedent`](@ref) method.

See also [`findbestantecedent`](@ref), [`BeamSearch`](@ref), [`RandSearch`](@ref).
"""
abstract type SearchMethod end

"""
    findbestantecedent(
        sm::SearchMethod,
        X::AbstractLogiset,
        y::AbstractVector{<:CLabel},
        w::AbstractVector;
        kwargs...
    )

Find the best antecedent formula using `sm` on dataset `X` labelled by `y` and weighted by `w`.

See also [`findbestantecedent`](@ref), [`SearchMethod`](@ref).
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


include("searchmethods/beamsearch.jl")
include("searchmethods/randsearch.jl")


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

Sort rule satmasks based on their loss, using a specified loss function.

Sorts rule antecedents based on their lossfnctn using a specified loss function.

Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *loss_function* function.
Then the permutation of the bests *beam_search* sorted antecedent is returned with the lossfnctn
value of the best one.

See also
[`entropy`](@ref).
"""
function sortantecedents(
    antecedents::AbstractVector{<:Tuple{Formula, SatMask}},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    loss_function::Function,
    min_rule_coverage::Integer,
    max_info_gain::Union{Real, Nothing},
    significance_alpha::Union{Real, Nothing};
    kwargs...
)::Tuple{AbstractVector, <:Real}

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
    if !isnothing(max_info_gain)
        @assert (0 <= max_info_gain <= 1) "max_info_gain not in range [0,1]"

        minloss = (1 - max_info_gain) * loss_function(y, w; kwargs...)
        indexes = map(aq -> begin
                    (index, lossfnctn) = aq
                    (lossfnctn >= minloss) && index
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
