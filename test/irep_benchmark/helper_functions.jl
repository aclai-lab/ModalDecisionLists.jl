using ModalDecisionLists
using SoleModels: apply
using StatsBase
using Statistics
using Distributions
using Random


function model_wrapper(X, y, rng; kwargs...)
    irepstar(X, y; rng = rng, kwargs...)
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

    return Dict(
        :train_accuracy => mean(model_train_preds .== y_train),
        :test_accuracy => mean(model_test_preds .== y_test),
        :num_rules => total_num_rules,
        :num_literals_per_rule => avg_num_literals_per_rule
    )
end


using Printf

function print_statistics(results::Dict)
    println("\n\t" * "="^70)
    println("\t   STATISTICHE DETTAGLIATE (95% Confidence Interval & Std Dev)")
    println("\t" * "="^70)
    
    # Intestazione della tabella
    @printf("\t%-30s | %-8s | %-8s | %-15s\n", "Metrica", "Media", "Std Dev", "Margine (95%)")
    println("\t" * "-"^70)

    # Lista delle metriche da iterare per evitare codice ripetitivo
    metrics = [
        (:train_accuracy, "Training Accuracy"),
        (:test_accuracy,  "Testing Accuracy"),
        (:num_rules,      "Number of Rules"),
        (:num_literals_per_rule, "Literals per Rule")
    ]

    for (key, label) in metrics
        m = results[key].mean
        s = results[key].std
        err = results[key].margin
        
        @printf("\t%-30s | %8.4f | %8.4f | ± %-8.4f\n", label, m, s, err)
    end
    
    println("\t" * "="^70 * "\n")
end


# function print_statistics(results::Dict)
#     println("\t=== 95% confidence intervals ===")
#     println("\t\tTraining accuracy: $(round(results[:train_accuracy].mean; digits=4)) ± $(round(results[:train_accuracy].margin, digits=4))")
#     println("\t\tTesting accuracy: $(round(results[:test_accuracy].mean; digits=4)) ± $(round(results[:test_accuracy].margin, digits=4))")

#     println("\t\tNumber of rules: $(round(results[:num_rules].mean; digits=4)) ± $(round(results[:num_rules].margin, digits=4))")
#     println("\t\tNumber of literals per rule: $(round(results[:num_literals_per_rule].mean; digits=4)) ± $(round(results[:num_literals_per_rule].margin, digits=4))")
# end