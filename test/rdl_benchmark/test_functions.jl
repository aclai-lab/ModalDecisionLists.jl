using DataFrames
using SoleModels
using SoleModels: apply, DecisionList, solemodel, info, models, nmodels, weighted_aggregation, rulebase, nleaves, height, listrules
using SoleData
using SoleData: AbstractInterpretationSet, AbstractInterpretation
using MLJ
using CategoricalArrays: CategoricalValue, CategoricalArray, Continuous
using RDatasets
using StatsBase
using Statistics
using Distributions
using Random
using ModalDecisionLists
using ScientificTypes: Continuous

DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree


function rdl_tree_wrapper(X, y, w; rng, kwargs...)
    num_features = nfeatures(X)

    # Prepare data for MLJ
    X_fixed = DataFrame(X)  
    y_fixed = categorical(y) 

    # X_fixed = coerce(X_fixed, Textual => OrderedFactor)


    # TODO: actually implement this into machine()
    actual_w = isnothing(w) ? nothing : collect(Float64, w)
    n_sub = round(Integer, sqrt(num_features)) + 2

    tree_model = DecisionTreeClassifier(rng = rng, 
                                        n_subfeatures = n_sub, 
                                        post_prune = false,   # Explicitly disable post-pruning
                                        max_depth = -1        # -1 means no limit to depth
                                        )
    
    mach = machine(tree_model, X_fixed, y_fixed)

    fit!(mach, verbosity = 0)
    
    trained_features = report(mach).features        # set of features from X

    sole_tree = solemodel(fitted_params(mach).tree; featurenames = trained_features)

    return sole_tree    
end


function split_model_wrapper(X, y, w; rng, iteration, num_lists, method_used = :sequentialcovering, kwargs...)
    if iteration <= num_lists
        X_plain = DataFrame(X)

        for col in names(X_plain)
            if eltype(X_plain[!, col]) <: CategoricalValue
                X_plain[!, col] = String.(X_plain[!, col])
            end
        end

        X_plain = PropositionalLogiset(X_plain)

        if method_used == :sequentialcovering
            return sequentialcovering(X_plain, y, w; rng = rng, kwargs...)
        elseif method_used == :irepstar
            return irepstar(X_plain, y, w; rng = rng, kwargs...)
        elseif method_used == :ripperk || method_used == :ripper
            return ripperk(X_plain, y, w; max_k = 2, rng = rng, kwargs...)
        end
        
        throw(ArgumentError("Method not implemented"))
    else
        return rdl_tree_wrapper(X, y, w; rng = rng, kwargs...)  
    end 

end

# Function passed to repeated_cv to train an ensemble model with a number 'num_lists' of DecisionLists, and 'num_models - num_lists' of decision tree
function rdl_model_wrapper(X, y, rng; num_models, num_lists, kwargs...)
    num_features = nfeatures(X)

    samples_ratio_per_model = (num_models == 1) ? 1.0 : 0.7
    n_subfeatures_per_model = num_features
    use_bootstrapping = (num_models != 1)

    rdl_ensemble = build_ensemble(
        X, 
        y, 
        num_models;
        
        # rdl arguments
        use_bootstrapping = use_bootstrapping,
        samples_ratio_per_model = samples_ratio_per_model,
        n_subfeatures_per_model = n_subfeatures_per_model,
        rng = rng,
        model_wrapper = split_model_wrapper,

        # kwargs passed directly to split_model_wrapper
        num_lists = num_lists,
        kwargs...
    )

    return rdl_ensemble
end


function __apply_post(m, preds)
    if haskey(info(m), :apply_postprocess)
        apply_postprocess_f = info(m, :apply_postprocess)
        preds = apply_postprocess_f.(preds)
    end
    preds
end

function apply_dl(
    m::DecisionList,
    i::AbstractInterpretation;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    kwargs...
)
    for rule in rulebase(m)
        if checkantecedent(rule, i, check_args...; check_kwargs...)
            return consequent(rule)
        end
    end
    defaultconsequent(m)
end

function apply_dl(
    m::DecisionList{O},
    d::AbstractInterpretationSet;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    kwargs...
) where {O}
    nsamp = ninstances(d)
    preds = Vector{O}(undef, nsamp)
    uncovered_idxs = 1:nsamp

    for rule in rulebase(m)
        length(uncovered_idxs) == 0 && break

        uncovered_d = slicedataset(d, uncovered_idxs; return_view = true)

        idxs_sat = findall(
            checkantecedent(rule, uncovered_d, check_args...; check_kwargs...)
        )
        idxs_sat = uncovered_idxs[idxs_sat]
        uncovered_idxs = setdiff(uncovered_idxs, idxs_sat)

        foreach((i)->(preds[i] = outcome(consequent(rule))), idxs_sat)
    end

    length(uncovered_idxs) != 0 &&
        foreach((i)->(preds[i] = outcome(defaultconsequent(m))), uncovered_idxs)

    return preds
end

function apply_ensemble(
    m::DecisionEnsemble,
    X::PropositionalLogiset;
    suppress_parity_warning = false,
    kwargs...
)   
    submodels = models(m)

    total_preds = []

    for subm ∈ submodels
        if hasproperty(subm, :info) && haskey(info(subm), :featurenames)
            feature_names = info(subm)[:featurenames]

            # TODO: this is unsafe as 'PropositionalLogiset' does not necessarily allow slicing in this manner.
            # However, until PropositionalLogiset is reworked this must suffice
            X_model = X[:, feature_names]
            preds = if isa(subm, DecisionList) 
                apply_dl(subm, X_model; suppress_parity_warning = suppress_parity_warning, kwargs...)
            else
                apply(subm, X_model)
            end
        else
            preds = if isa(subm, DecisionList) 
                apply_dl(subm, X; suppress_parity_warning = suppress_parity_warning, kwargs...)
            else
                apply(subm, X)
            end
        end

        push!(total_preds, preds)
    end

    preds = hcat(total_preds...)
    preds = __apply_post(m, preds)
    preds = [
        weighted_aggregation(m)(preds[i,:]; suppress_parity_warning)
        for i in 1:size(preds,1)
    ]
    return preds
end


function compute_metrics(model, X_train, y_train, X_test, y_test; kwargs...)
    model_train_preds = apply_ensemble(model, X_train; suppress_parity_warning = true)
    model_test_preds = apply_ensemble(model, X_test, suppress_parity_warning = true)

    train_accuracy = mean(model_train_preds .== y_train)
    test_accuracy = mean(model_test_preds .== y_test)

    metrics = Dict(
        :train_accuracy => train_accuracy,
        :test_accuracy => test_accuracy
    )

    if isa(model, DecisionEnsemble)
        total_num_rules = 0
        total_num_connectives = 0

        for model ∈ models(model)
            if isa(model, DecisionList)
                model_rulebase = rulebase(model)

                total_num_rules += length(model_rulebase)

                for rule ∈ model_rulebase
                    total_num_connectives += 1 + nconnectives(rule.antecedent)
                end


            elseif isa(model, DecisionTree)
                total_num_rules += nleaves(model)
                total_num_connectives += total_path_connectives(model)
            end
            

        end

        avg_num_rules = total_num_rules / nmodels(model)
        avg_num_connectives_per_rule = total_num_connectives / total_num_rules

        metrics[:avg_num_rules] = avg_num_rules
        metrics[:avg_num_connectives_per_rule] = avg_num_connectives_per_rule
    end

    return metrics
end


function total_path_connectives(model::AbstractModel)
    if model isa LeafModel
        return 0
    elseif model isa Branch
        # Each branch node contributes 1 connective (∧) to every rule
        # that descends through it. That equals the number of leaves below it.
        left  = posconsequent(model)
        right = negconsequent(model)
        n_leaves_left  = count_leaves(left)
        n_leaves_right = count_leaves(right)
        return (n_leaves_left + n_leaves_right) +           # this branch's contribution
               total_path_connectives(left) +
               total_path_connectives(right)
    elseif model isa DecisionTree
        return total_path_connectives(root(model))
    else
        error("Unexpected model type: $(typeof(model))")
    end
end

function count_leaves(model::AbstractModel)
    if model isa LeafModel
        return 1
    elseif model isa Branch
        return count_leaves(posconsequent(model)) + count_leaves(negconsequent(model))
    elseif model isa DecisionTree
        return count_leaves(root(model))
    else
        error("Unexpected model type: $(typeof(model))")
    end
end