
############################################################################################
################### SequentialCovering - DecisionSet #######################################
############################################################################################

function sequentialcovering_unordered(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{U},Symbol}=default_weights(length(y));
    searchmethod::SearchMethod=BeamSearch(),
    unorderedstrategy::Bool = false,
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    min_rule_coverage::Integer=1,
    suppress_parity_warning::Bool=true,
    kwargs...
)::DecisionList where {U<:Real}

    !isnothing(max_rulebase_length) && @assert max_rulebase_length > 0 "`max_rulebase_length` must be  > 0"
    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]

    w = if isnothing(w) || w == :default
        default_weights(y)
    elseif w == :rebalance
        balanced_weights(y)
    else
        w
    end

    searchmethod = reconstruct(searchmethod,  kwargs)

    !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
    !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")
    (ninstances(X) == 0) && error("Empty trainig set")

    y, labels = y |> maptointeger

    nlabels = labels |> length

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]
    for target_class in 1:nlabels

        newrules = find_rules(
            searchmethod,
            uncoveredX,
            uncoveredy,
            uncoveredw;
            target_class,
            nlabels
        )
        if !isnothing(max_rulebase_length) && length(rulebase) > (max_rulebase_length - 1)
            break
        end
    end

    defaultconsequent = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    return DecisionList(rulebase, labels[defaultconsequent])
end
