using SoleBase
using SoleBase: CLabel
using SoleLogics
using SoleLogics: nconjuncts, pushconjunct!
using SoleData
import SoleData: ScalarCondition, PropositionalLogiset, AbstractAlphabet, UnionAlphabet
import SoleData: alphabet, test_operator, isordered, polarity, atoms
using SoleModels
using SoleModels: DecisionList, Rule, ConstantModel
using SoleModels: default_weights, balanced_weights, bestguess
using DataFrames
using StatsBase: mode, countmap, counts, Weights
using FillArrays
using ModalDecisionLists

############################################################################################
############## Sequential Covering #########################################################
############################################################################################

"""
    function sequentialcovering(
        X::PropositionalLogiset,
        y::AbstractVector{<:CLabel},
        w::Union{Nothing,AbstractVector{U},Symbol} = default_weights(length(y));
        search_method::SearchMethod=BeamSearch(),
        max_rulebase_length::Union{Nothing,Integer}=nothing,
        suppress_parity_warning::Bool=false,
        kwargs...
    )::DecisionList where {U<:Real}

Learn a decision list on an logiset `X` with labels `y` and weights `w` following
the classic [sequential covering](https://christophm.github.io/interpretable-ml-book/rules.html#sequential-covering) learning scheme.
This involves iteratively learning a single rule, and removing the newly covered instances.

# Arguments
TODO

# Examples
TODO

See also [`TODO`](@ref), [`PropositionalLogiset`](@ref), [`BeamSearch`](@ref).

"""
function sequentialcovering(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{U},Symbol}=default_weights(length(y));
    searchmethod::SearchMethod=BeamSearch(),
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    suppress_parity_warning::Bool=false,
    kwargs...
)::DecisionList where {U<:Real}

    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]

    w = if isnothing(w) || w == :default
        default_weights(y)
    elseif w == :rebalance
        balanced_weights(y)
    else
        w
    end

    !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
    !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")

    (ninstances(X) == 0) && error("Empty trainig set")

    y, labels = y |> maptointeger

    # DEBUG
    # uncoveredslice = collect(1:ninstances(X))

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]
    while true
        bestantecedent, bestantecedent_coverage = findbestantecedent(
            searchmethod,
            uncoveredX,
            uncoveredy,
            uncoveredw;
            kwargs...
        )
        bestantecedent == ⊤ && break

        rule = begin
            justcoveredy = uncoveredy[bestantecedent_coverage]
            justcoveredw = uncoveredw[bestantecedent_coverage]
            consequent = labels[bestguess(justcoveredy, justcoveredw)]
            info_cm = (;
                supporting_labels=collect(justcoveredy),
                supporting_weights=collect(justcoveredw)
            )
            consequent_cm = ConstantModel(consequent, info_cm)
            Rule(bestantecedent, consequent_cm)
        end

        push!(rulebase, rule)

        uncoveredX = slicedataset(uncoveredX, (!).(bestantecedent_coverage); return_view=true)
        uncoveredy = @view uncoveredy[(!).(bestantecedent_coverage)]
        uncoveredw = @view uncoveredw[(!).(bestantecedent_coverage)]

        if !isnothing(max_rulebase_length) && length(rulebase) > max_rulebase_length
            break
        end

        # setdiff!(uncoveredslice, uncoveredslice[bestantecedent_coverage])
        # uncoveredX = slicedataset(X, uncoveredslice; return_view = true)
        # uncoveredy = @view y[uncoveredslice]
        # uncoveredw = @view w[uncoveredslice]
    end

    # !allequal(uncoveredy) && @warn "Remaining classes are not all equal; defaultclass represents the best estimate."

    defaultconsequent = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    return DecisionList(rulebase, defaultconsequent)
end

function sole_cn2(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=BeamSearch(), kwargs...)
end

function sole_rand(
    X::PropositionalLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=RandSearch(), kwargs...)
end

#= Int.(values(currentrule_distribution)) =#
# currentrule_distribution = Dict(unique(y) .=> 0)

# for c in coveredy
#     currentrule_distribution[c] += 1
# end
