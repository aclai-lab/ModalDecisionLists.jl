module LossFunctions

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions
using ..ModalDecisionLists: Antecedent
using ..Metrics

############################################################################################
############################# New Loss Functions ###########################################
############################################################################################


# A top-level abstract type for all loss functions
abstract type AbstractLossFunction end

# Subtypes to group the functions by symmetry
abstract type SymmetricLoss <: AbstractLossFunction end
abstract type AsymmetricLoss <: AbstractLossFunction end

# export AbstractLossFunction
# export SymmetricLoss
# export AsymmetricLoss

function calculate_loss(::AbstractLossFunction; kwargs...)
    error("calculate_loss can only be called with a non-abstract loss function type")
end

#####################################################
################# SYMMETRIC LOSSES ##################
#####################################################

struct GiniImpurity <: SymmetricLoss end

function (::GiniImpurity)(
    y::AbstractVector{<:Integer},
    w::AbstractVector = default_weights(length(y))
)
    return gini_impurity(y, w)                      
end


struct Entropy <: SymmetricLoss end

function (::Entropy)(
    y::AbstractVector{<:Integer},
    w::AbstractVector=default_weights(length(y));
    kwargs...
)
    return entropy(y, w; kwargs...)                 
end


struct LaplaceMetric <: SymmetricLoss end

function (::LaplaceMetric)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector=default_weights(length(y));
    nlabels::Integer,
    kwargs...
)
    return laplace_metric(y, w; nlabels, kwargs...)         
end

#####################################################
################# ASYMMETRIC LOSSES #################
#####################################################


struct FOILGain <: AsymmetricLoss end

function (::FOILGain)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector,
    target_class::Integer;
    antecedent::Antecedent = nothing,
    prev_antecedent::Antecedent = nothing,
    nlabels::Integer,
    kwargs...
)
    # In information gain, se prev_antecedent = nothing, si può
    #   A. Assumere che gain = ∞, tuttavia così l'algoritmo non ritornerà mai ⊤ e bisogna considerare come opera
    #       la funzione di stopping di IREP*
    #   B. Porre gain = 0, come fa in wittgenstein (base_functions.py linea 538) (MIGLIORE)
    if isnothing(prev_antecedent)
        return 0 end


    # 1 dove y è uguale a terget_class 
    target_vector = (y .== target_class) .> 0   # NOTE: the .> 0 Converts this to a BitVector

    # TODO: Tenere conto dei pesi
    tp1 = sum(antecedent.covmask .& target_vector)         
    fp1 = sum(antecedent.covmask .& (.!target_vector))

    tp0 = sum(prev_antecedent.covmask .& target_vector)
    fp0 = sum(prev_antecedent.covmask .& (.!target_vector))

    # precision
    prec1 = tp1 / (tp1 + fp1)
    prec0 = tp0 / (tp0 + fp0)
    t = sum(prev_antecedent.covmask .& antecedent.covmask)  # sample coperti da entrambi

    return t * ( log2(prec1) - log2(prec0) )
end


struct LaplaceAccuracy <: AsymmetricLoss end    

function (::LaplaceAccuracy)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector,
    target_class::Integer;
    antecedent::Antecedent = nothing,
    prev_antecedent::Union{Antecedent, Nothing} = nothing,
    nlabels::Integer,
    kwargs...
)
    # 1 dove y è pari a terget_class, 0 altrimenti
    target_vector = (y .== target_class) .> 0   # NOTE: the .> 0 Converts this to a BitVector

    # TODO: Tenere conto dei pesi
    # tp = true positive, fp = false positive
    tp = sum(antecedent.covmask .& target_vector)         
    fp = sum(antecedent.covmask .& (.!target_vector))

    return 1 - (tp + 1) / (tp + fp + 2)
end

end
