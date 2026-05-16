using SoleBase: CLabel
using SoleData: AbstractLogiset, PropositionalLogiset
using SoleModels: bestguess
using Parameters
using FillArrays
using StatsBase
using .LossFunctions


include("searchmethods/beamsearch.jl")
include("searchmethods/randsearch.jl")
include("searchmethods/atom-generator.jl")
include("searchmethods/random-generator.jl")



"""
    TODO sortantecedents....


Takes an *antecedents*, each decorated by a SatMask indicating his coverage bitmask.
Each antecedent is evaluated on his covered y using the provided *loss_function* function.
Then the permutation of the bests *beam_width* sorted antecedent is returned with the lossfnctn
value of the best one.

See also
[`entropy`](@ref).
"""
function sortantecedents(
    antecedents::AbstractVector{Antecedent},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    loss_function::LossFunctions.SymmetricLoss,
    min_rule_coverage::Integer,
    max_infogain_ratio::Union{Real,Nothing},
    significance_alpha::Union{Real,Nothing};
    kwargs...
)::Tuple{AbstractVector,<:Real}
    isempty(antecedents) && return [], Inf

    # If 'min_rule_coverage' is defined, this filters out from antecedents any antecedent whose covmasks covers less than 'min_rule_coverage' samples 
    if min_rule_coverage > 1
        validindices = findall(ant -> count(ant.covmask) >= min_rule_coverage, antecedents)
        isempty(validindices) && return [], Inf
        antecedents = antecedents[validindices]
    end

    indices = eachindex(antecedents)

    # loss function values for each antecedent
    antslossfnctn = map(a ->  loss_function(y, w; antecedent=a, kwargs...) , antecedents)

    if !isnothing(max_infogain_ratio)
        # every rule whose loss is < const. * loss of ⊤ over dataset is to be removed, this makes the actual sorting at the end faster
        # and removes antecedents that may be overfitting
        bot_ant = bot_antecedent(length(y))
        minloss = (1-max_infogain_ratio) * loss_function(y, w; antecedent = bot_ant, kwargs...)

        # Keep only the indices corresponding antecedents whose loss is ≥ min_loss
        indices = [ind for (ind, loss) in enumerate(antslossfnctn) if loss ≥ minloss]

        isempty(indices) && return [], Inf
    end

    # Extract the indices (with respect to antslossfnctn) of the 'beam_width' best antecedents (with lowest loss)
    valid_indices = partialsortperm(antslossfnctn[indices], 1:min(beam_width, length(indices)))

    newstar_perm = indices[valid_indices]  # convert indices to those relative to the parameter antecedents
    newstar = antecedents[newstar_perm]    # extract best antecedents and corresponding loss functions
    bestantecedent_lossfnctn = antslossfnctn[newstar_perm[1]]

    return newstar, bestantecedent_lossfnctn
end





function sortantecedents(
    antecedents_with_parents::AbstractVector{<:Tuple{Antecedent, Union{Nothing, Antecedent}}},
    y::AbstractVector{<:CLabel},
    w::AbstractVector,
    beam_width::Integer,
    loss_function::LossFunctions.AsymmetricLoss,
    min_rule_coverage::Integer,
    max_infogain_ratio::Union{Real,Nothing},
    significance_alpha::Union{Real,Nothing};
    target_class::Union{Integer, Nothing},
    kwargs...
)::Tuple{AbstractVector,<:Real} 
    isempty(antecedents_with_parents) && return [], Inf

    # If 'min_rule_coverage' is defined, this filters out from antecedents any antecedent whose covmasks covers less than 'min_rule_coverage' samples 
    if min_rule_coverage > 1
        validindices = findall(tup -> count(tup[1].covmask) >= min_rule_coverage, antecedents_with_parents)
        isempty(validindices) && return Antecedent[], Inf
        antecedents_with_parents = antecedents_with_parents[validindices]
    end

    indices = eachindex(antecedents_with_parents)

    # loss function values for each antecedent
    antslossfnctn = map(antecedents_with_parents) do (child_ant, parent_ant)
        # prev_antecedent is the parent to the loss_function, this is necessary if the loss function is a "delta_loss"
        loss_function(y, w, target_class; antecedent=child_ant, prev_antecedent=parent_ant, kwargs...)
    end

    if !isnothing(max_infogain_ratio)
        # every rule whose loss is < const. * loss of ⊤ over dataset is to be removed, this makes the actual sorting at the end faster
        # and removes antecedents that may be overfitting
        bot_ant = bot_antecedent(length(y))
        minloss = (1-max_infogain_ratio) * loss_function(y, w, target_class; antecedent = bot_ant, kwargs...)

        # Keep only the indices corresponding antecedents whose loss is ≥ min_loss
        indices = [ind for (ind, loss) in enumerate(antslossfnctn) if loss ≥ minloss]

        isempty(indices) && return [], Inf
    end

    # Extract the indices (with respect to antslossfnctn) of the 'beam_width' best antecedents (with lowest loss)
    valid_indices = partialsortperm(antslossfnctn[indices], 1:min(beam_width, length(indices)))

    newstar_perm = indices[valid_indices]  # convert indices to those relative to the parameter antecedents
    newstar = [antecedents_with_parents[idx][1] for idx in newstar_perm]    # extract best antecedents and corresponding loss functions
    bestantecedent_lossfnctn = antslossfnctn[newstar_perm[1]]

    return newstar, bestantecedent_lossfnctn
end