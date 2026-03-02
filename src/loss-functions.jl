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


#####################################################
################# SYMMETRIC LOSSES ##################
#####################################################

struct GiniImpurity <: SymmetricLoss end

function (::GiniImpurity)(
    y::AbstractVector{<:Integer},
    w::AbstractVector{<:Real} = default_weights(length(y))
)
    return gini_impurity(y, w)                      
end


struct Entropy <: SymmetricLoss end

function (::Entropy)(
    y::AbstractVector{<:Integer},
    w::AbstractVector{<:Real}=default_weights(length(y));
    kwargs...
)
    return entropy(y, w; kwargs...)                 
end


struct LaplaceMetric <: SymmetricLoss end

function (::LaplaceMetric)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector{<:Real}=default_weights(length(y));
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
    w::AbstractVector{<:Real},
    target_class::Integer;
    antecedent::Union{Antecedent, Nothing} = nothing,
    prev_antecedent::Union{Antecedent, Nothing} = nothing,
    kwargs...
)
    # TODO: This function should probably throw an error if both antecedent and prev_antecedent are null. This, however, depends on 
    # where and how we actually want to use this outside of findbestantecedent
    # TODO: Every ".&" operation creates a temporary vector in memory. We should consider replacing everything with a loop over y and
    # manually counting true positives and false positives for both antecedents. Given that the whole thing is pre-compiled, this might
    # actually speed things up and it would certainly reduce memory allocation.
    
    # If there is no previous antecedent, we return 0.
    if isnothing(prev_antecedent) || isnothing(antecedent)
        return 0 end

    # elements are 1 where y equals the terget_class 
    target_vector = (y .== target_class) .> 0   # NOTE: the .> 0 Converts this to a BitVector

    # calculate true and false positives for antecedent
    tp1_mask = antecedent.covmask .& target_vector          # mask is 1 if the sample is a true positive for antecedent, and zero otherwise
    fp1_mask = antecedent.covmask .& (.!target_vector)      # mask is 1 if the sample is a false positive for antecedent, and zero otherwise
    tp1 = sum(tp1_mask .* w)
    fp1 = sum(fp1_mask .* w)

    # calculate true and false positives fro prev_antecedent
    tp0_mask = prev_antecedent.covmask .& target_vector
    fp0_mask = prev_antecedent.covmask .& (.!target_vector)
    tp0 = sum(tp0_mask .* w)
    fp0 = sum(fp0_mask .* w)

    # precision
    prec_curr = (tp1 + fp1 > 0) ? tp1 / (tp1 + fp1) : 0.0       # make sure division by zero does not occurr
    prec_prev = (tp0 + fp0 > 0) ? tp0 / (tp0 + fp0) : 0.0
    
    simultaneous_cover_mask = prev_antecedent.covmask .& antecedent.covmask     # mask is 1 if the corresponding sample is covered by both antecedent and prev_antecedent
    t = sum(simultaneous_cover_mask .* w)  

    # A higher FOILGain value corresponds to better accuracy. Since this is to be treated as a loss function we must return 
    # the negative value of the actual Gain to make sure that better antecedent choices have a lower loss value when they have a higher information gain.
    return -t * ( log2(prec_curr) - log2(prec_prev) )       
end


struct LaplaceAccuracy <: AsymmetricLoss end    

function (::LaplaceAccuracy)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector{<:Real},
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
