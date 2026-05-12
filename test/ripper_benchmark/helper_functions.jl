using ModalDecisionLists
using SoleModels: apply
using StatsBase
using Statistics
using Distributions
using Random
using MLJ
using Printf

function model_wrapper(X, y, rng; kwargs...)
    ripperk(X, y; rng = rng, kwargs...)
end

function metrics_wrapper(model, X_train, y_train, X_test, y_test; kwargs...)
    model_train_preds = apply(model, X_train)
    model_test_preds = apply(model, X_test)

    total_num_literals = 0

    model_rules = rulebase(model)
    for rule ∈ model_rules
        total_num_literals += 1 + nconnectives(rule.antecedent)
    end
    
    total_num_rules = length(model_rules)
    avg_num_literals_per_rule = total_num_literals / total_num_rules

    # Cohen's kappa coefficient calculation
    target_levels = union(unique(y_test), unique(y_train), unique(model_train_preds))
    
    #   force the same underlying pool
    y_tr_cat, p_tr_cat = categorical(y_train), categorical(model_train_preds)
    y_te_cat, p_te_cat = categorical(y_test),  categorical(model_test_preds)
    foreach(v -> levels!(v, target_levels), (y_tr_cat, p_tr_cat, y_te_cat, p_te_cat))

    train_kappa = kappa(p_tr_cat, y_tr_cat)
    test_kappa  = kappa(p_te_cat, y_te_cat)

    return Dict(
        :train_accuracy => mean(model_train_preds .== y_train),
        :test_accuracy => mean(model_test_preds .== y_test),
        :num_rules => total_num_rules,
        :num_literals_per_rule => avg_num_literals_per_rule,
        :train_kappa => train_kappa,
        :test_kappa => test_kappa
    )
end



function print_statistics(results::Dict)
    println("\n\t" * "="^85)
    println("\t   Detailed Results (95% Confidence Interval & Std Dev)")
    println("\t" * "="^85)
    
    # Table headings
    @printf("\t%-30s | %-8s | %-8s | %-15s\n", "Metrica", "Media", "Std Dev", "Margine (95%)")
    println("\t" * "-"^85)

    # List of various metrics (ho aggiunto i Kappa)
    metrics = [
        (:train_accuracy, "Training Accuracy"),
        (:test_accuracy,  "Testing Accuracy"),
        (:train_kappa,    "Training Cohen's Kappa"),
        (:test_kappa,     "Testing Cohen's Kappa"),
        (:num_rules,      "Number of Rules"),
        (:num_literals_per_rule, "Literals per Rule")
    ]

    for (key, label) in metrics
        # verify if the key exists to avoid errors +
        if haskey(results, key)
            m = results[key].mean
            s = results[key].std
            err = results[key].margin
            
            @printf("\t%-30s | %8.4f | %8.4f | ± %-8.4f\n", label, m, s, err)
        end
    end
    
    println("\t" * "="^85 * "\n")
end
