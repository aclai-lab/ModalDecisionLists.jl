using SoleBase
using SoleBase: CLabel
using SoleLogics
using SoleLogics: nconjuncts, pushconjunct!
using SoleData
using SoleData: AbstractLogiset
import SoleData: ScalarCondition, PropositionalLogiset, AbstractAlphabet, UnionAlphabet
import SoleData: alphabet, test_operator, isordered, polarity, atoms
using SoleModels
using SoleModels: DecisionList, Rule, ConstantModel
using SoleModels: default_weights, balanced_weights, bestguess
using DataFrames
using StatsBase: mode, countmap, counts, Weights
using FillArrays
using ModalDecisionLists
using Parameters

################################################################################
############## Sequential Covering #############################################
################################################################################

"""
    function sequentialcovering(
        X::AbstractLogiset,
        y::AbstractVector{<:CLabel},
        w::Union{Nothing,AbstractVector{U},Symbol} = default_weights(length(y));
        kwargs...
    )::DecisionList where {U<:Real}

Learn a decision list on an logiset `X` with labels `y` and weights `w` following
the classic [sequential covering](https://christophm.github.io/interpretable-ml-book/rules.html#sequential-covering) learning scheme.
This involves iteratively learning a single rule, and removing the newly covered instances.

# Keyword Arguments

* `searchmethod::SearchMethod`: The search method for finding single rules (see [`SearchMethod`](@ref));
* `max_rulebase_length::Union{Nothing,Integer}` is the maximum length of the rulebase;
* `suppress_parity_warning::Bool` if `true`, suppresses parity warnings.
* `unorderedstrategy::Bool`: TODO @Edo explain
* `min_rule_coverage::Integer`: TODO @Edo explain
* Any additional keyword argument will be imputed to the `searchmethod`, replacing its original value.

# Examples

```julia-repl

julia> X = PropositionalLogiset(iris_dataframe)


julia> y = Vector{CLabel}(iris_labels)


julia> sequentialcovering(X, y)
▣
├[1/22]┐(:sepal_length ≤ 4.8)
│└ setosa
├[2/22]┐(:sepal_length ≥ 7.1)
│└ virginica
├[3/22]┐(:sepal_length ≥ 7.0)
│└ versicolor
├[4/22]┐(:sepal_width ≤ 2.0)
│└ versicolor
├[5/22]┐(:sepal_width ≥ 3.5)
│└ setosa
├[6/22]┐(:petal_length ≤ 1.7)
│└ setosa
├[7/22]┐(:petal_length ≤ 4.4)
│└ versicolor
├[8/22]┐(:sepal_length ≤ 4.9)
│└ virginica
├[9/22]┐(:sepal_length ≤ 5.4)
│└ versicolor
├[10/22]┐(:petal_length ≤ 4.7)
│└ versicolor
├[11/22]┐(:sepal_length ≤ 5.8)
│└ virginica
├[12/22]┐(:sepal_width ≤ 2.2)
│└ virginica
├[13/22]┐(:sepal_width ≥ 3.3)
│└ virginica
├[14/22]┐(:petal_length ≥ 5.2)
│└ virginica
├[15/22]┐(:petal_width ≤ 1.4)
│└ versicolor
├[16/22]┐(:petal_width ≥ 1.9)
│└ virginica
├[17/22]┐(:sepal_length ≥ 6.7)
│└ versicolor
├[18/22]┐(:sepal_width ≤ 2.5)
│└ versicolor
├[19/22]┐(:sepal_length ≥ 6.1)
│└ virginica
├[20/22]┐(:sepal_width ≤ 2.7)
│└ versicolor
├[21/22]┐(:sepal_length ≥ 6.0)
│└ virginica
├[22/22]┐(:sepal_width ≤ 3.0)
│└ virginica
└✘ versicolor
```

See also
[`SearchMethod`](@ref), [`BeamSearch`](@ref), [`PropositionalLogiset`](@ref), [`DecisionList`](@ref).
"""
function sequentialcovering(
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

    searchmethod = reconstruct(searchmethod, kwargs)

    !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
    !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")
    (ninstances(X) == 0) && error("Empty trainig set")

    y, labels = y |> maptointeger

    n_labels = labels |> length

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
            min_rule_coverage = min_rule_coverage,
            n_labels = n_labels
        )
        bestantecedent == ⊤ && break

        rule, consequent_i = begin
            justcoveredy = uncoveredy[bestantecedent_coverage]
            justcoveredw = uncoveredw[bestantecedent_coverage]
            consequent_i = SoleModels.bestguess(justcoveredy, justcoveredw; suppress_parity_warning=suppress_parity_warning)
            info_cm = (;
                # TODO @Italian anche qui c'è da cambiare qualcosa per il caso di DL non ordinata ( forse )
                supporting_labels=collect(justcoveredy),
                supporting_weights=collect(justcoveredw)
            )
            consequent_cm = ConstantModel(labels[consequent_i], info_cm)
            #
            (Rule(bestantecedent, consequent_cm), consequent_i)
        end
        push!(rulebase, rule)
        # TODO @Italian Attenzione, la DecisionList risultante non è applicabile con apply in quanto
        # non vale il metodo di applicazione della prima regola vera !!!!!!!!!!!!!!!!!!!!!!
        uncovered_slice = begin
            if unorderedstrategy
                correctclass_coverage = (uncoveredy .== consequent_i) .& bestantecedent_coverage
                (!).(correctclass_coverage)
            else
                (!).(bestantecedent_coverage)
            end
        end

        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]

        if !isnothing(max_rulebase_length) && length(rulebase) > (max_rulebase_length - 1)
            break
        end
    end

    # !allequal(uncoveredy) && @warn "Remaining classes are not all equal; defaultclass represents the best estimate."

    defaultconsequent = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    return DecisionList(rulebase, labels[defaultconsequent])
end

function build_cn2(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=BeamSearch(), kwargs...)
end

function build_orange_cn2(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    error("TODO: what's the default parametrization for orange CN2?")
    # return sequentialcovering(X, y, w; searchmethod=BeamSearch(), kwargs...)
end

function build_randcn2(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=RandSearch(), kwargs...)
end

