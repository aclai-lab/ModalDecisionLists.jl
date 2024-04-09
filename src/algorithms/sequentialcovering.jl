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
using Parameters

############################################################################################
############## Sequential Covering #########################################################
############################################################################################


# TODO @Edo ricontrollare docu
# devo dare alre indicazioni su X, y,w ?
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

# OptionalArguments

* `search_method::SearchMethod`: Search method for finding single rules.
* `max_rulebase_length` is the aximum length of the rulebase.
* `suppress_parity_warning` if true, suppresses parity warnings.

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
[`SearchMethod`](@ref), [`PropositionalLogiset`](@ref), [`DecisionList`](@ref).

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

    # Non funziona, lascia  i campi di searchmethod invariati (ma non ritorna errore)

    searchmethod = reconstruct(searchmethod,  kwargs)

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
        # TODO è una soluzione accettabile per il passaggio dei parametri ?
        bestantecedent, bestantecedent_coverage = findbestantecedent(
            searchmethod,
            uncoveredX,
            uncoveredy,
            uncoveredw
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