module LossFunctions

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
using Distributions
using ..ModalDecisionLists: Antecedent, extract_covered_labels
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
    w::AbstractVector{<:Real} = default_weights(length(y));
    antecedent::Union{Antecedent, Nothing} = nothing,
    kwargs...
)
    y_covered, w_covered =  if !isnothing(antecedent)
        extract_covered_labels(antecedent, y, w)
    else
        y, w
    end
    return Metrics.gini_impurity(y_covered, w_covered)            
end

struct Entropy <: SymmetricLoss end

function (::Entropy)(
    y::AbstractVector{<:Integer},
    w::AbstractVector{<:Real}=default_weights(length(y));
    antecedent::Union{Antecedent, Nothing} = nothing,
    kwargs...
)
    y_covered, w_covered =  if !isnothing(antecedent)
        extract_covered_labels(antecedent, y, w)
    else
        y, w
    end

    return Metrics.entropy(y_covered, w_covered; kwargs...)                 
end


struct LaplaceMetric <: SymmetricLoss end

function (::LaplaceMetric)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector{<:Real}=default_weights(length(y));
    antecedent::Union{Antecedent, Nothing} = nothing,
    nlabels::Integer,
    kwargs...
)
    y_covered, w_covered =  if !isnothing(antecedent)
        extract_covered_labels(antecedent, y, w)
    else
        y, w
    end

    # Since the goal is to maximize the laplace metric, in order to turn this into a minimization problem we must
    # minimize the corresponding error, which is equal to 1 - laplace_metric
    return 1 - Metrics.laplace_metric(y_covered, w_covered; nlabels, kwargs...)
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
    if isnothing(prev_antecedent) && isnothing(antecedent)
        throw(ArgumentError("`antecedent`and `prev_antecedent` cannot both be nothing")) end

+    # If there is no antecedent or previous antecedent we can't compare anything, we return 0 as there is no information gain to possibly be calculated.
    if isnothing(prev_antecedent) || isnothing(antecedent)
        return 0 end

    # Every ".&" operation creates a temporary vector in memory. Handling everything with a loop over y and
    # manually counting true positives and false positives for both antecedents is more efficient than using broadcasting

    n_samples = length(y)
    tp1 = 0.0; tp0 = 0.0;
    fp1 = 0.0; fp0 = 0.0;
    t = 0.0;

    # Manually loop through all samples and check for true/false positivies and true/false negatives for each class
    for i = 1 : n_samples
        # if sample is covered by first antecedent
        if antecedent.covmask[i]
            # if class is the target one this is a true positive, otherwise it's a false positive
            if y[i] == target_class     
                tp1 += w[i];
            else
                fp1 += w[i];
            end
        end

        if prev_antecedent.covmask[i]
            if y[i] == target_class
                tp0 += w[i];
            else
                fp0 += w[i];
            end
        end

        if antecedent.covmask[i] && prev_antecedent.covmask[i] && y[i] == target_class
            t += w[i];
        end

    end

    prec_curr = (tp1 + fp1 > 0) ? tp1 / (tp1 + fp1) : 0.0;       # make sure division by zero does not occurr
    prec_prev = (tp0 + fp0 > 0) ? tp0 / (tp0 + fp0) : 0.0;

    # handle edge cases which would cause the return type fo be NaN (ex: Inf - Inf or 0 * Inf)
    t == 0 && return 0.0
    prec_curr == 0 && return Inf
    prec_prev == 0 && return -Inf   # depends on convention, but usually safe

    return -t * ( log2(prec_curr) - log2(prec_prev) );             
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
    # no antecedent means no accuracy
    isnothing(antecedent) && return 0.0

    # 1 where y is equal to target_class, 0 otherwise
    target_vector = (y .== target_class)

    # tp = true positive, fp = false positive
    tp_mask = antecedent.covmask .& target_vector
    fp_mask = antecedent.covmask .& (.!target_vector)

    tp = sum(w .* tp_mask)         
    fp = sum(w .* fp_mask)

    return 1 - (tp + 1) / (tp + fp + 2)
end


struct MEstimate <: AsymmetricLoss end


function (::MEstimate)(
    y::AbstractVector{<:UInt32},
    w::AbstractVector{<:Real},
    target_class::Integer;
    antecedent::Antecedent = nothing,
    prev_antecedent::Union{Antecedent, Nothing} = nothing,
    nlabels::Integer,
    m::Real = 2.0, # The 'm' smoothing parameter
    prior::Union{Nothing, Real} = nothing,
    kwargs...
)
    # No antecedent means no accuracy (or infinite loss)
    isnothing(antecedent) && return 1.0

    # 1 where y is equal to target_class, 0 otherwise
    target_vector = (y .== target_class)

    # tp = true positive, fp = false positive
    tp_mask = antecedent.covmask .& target_vector
    fp_mask = antecedent.covmask .& (.!target_vector)

    tp = sum(w .* tp_mask)         
    fp = sum(w .* fp_mask)
    n = tp + fp # Total coverage of the rule

    # calculate prior probability from the data if not provided
    # Usually the frequency of target_class in the whole dataset
    if isnothing(prior)
        prior = sum(w .* target_vector) / sum(w)
    end

    # M-estimate formula: (tp + m * prior) / (n + m)
    accuracy = (tp + m * prior) / (n + m)

    # Return as loss (1 - accuracy)
    return 1.0 - accuracy
end




end
