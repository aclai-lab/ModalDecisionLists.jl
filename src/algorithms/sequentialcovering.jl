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
using SoleModels: default_weights, balanced_weights, bestguess, checkantecedent, rulebase
using DataFrames
using StatsBase: mode, countmap, counts, Weights
using FillArrays
using ModalDecisionLists
using Parameters
using Random

############################################################################################
################### SequentialCovering - DecisionList ######################################
############################################################################################
# * `unorderedstrategy::Bool`: TODO @Edo explain
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
* `loss_function::Function = ModalDecisionLists.Metrics.entropy` is the function that assigns a score to each partial solution.
* `max_infogain_ratio::Real=1.0`: constrains the maximum information gain for anantecedent with respect to the uncovered training set. Its value is bounded between 0 and 1.
* `default_alphabet::Union{Nothing,AbstractAlphabet}=nothing` offers the flexibility to define a tailored alphabet upon which antecedents generation occurs.
* `discretizedomain::Bool=false`:  discretizes continuous variables by identifying optimal cut points
* `significance_alpha::Union{Real,Nothing}=0.0` is the significant alpha
* `min_rule_coverage::Union{Nothing,Integer} = 1` specifies the minimum number of instances covered by each rule.
* `max_rule_length::Union{Nothing,Integer} = nothing` specifies the maximum length allowed for a rule in the search algorithm.

# Examples

```julia-repl

julia> X = PropositionalLogiset(iris_dataframe);

julia> y = Vector{CLabel}(iris_labels);

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
│└ virginicaw::Union{Nothing,AbstractVector{Real},Symbol}
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
    # Da incapsulaper dentro InstanceSet
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{U},Symbol}=default_weights(length(y)); 
    searchmethod::SearchMethod=BeamSearch(), 
    # loss_function::Function=ModalDecisionLists.Metrics.entropy,
    loss_function::ModalDecisionLists.LossFunctions.AsymmetricLoss = ModalDecisionLists.LossFunctions.Entropy(),
    max_infogain_ratio::Real=1.0, 
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    discretizedomain::Bool=false,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Integer=1,
    max_rule_length::Union{Nothing,Integer}=nothing,
    max_rulebase_length::Union{Nothing,Integer}=nothing,
    suppress_parity_warning::Bool=false,
    kwargs...
)::DecisionList where {U<:Real}

    !isnothing(max_rulebase_length) && @assert max_rulebase_length > 0 "`max_rulebase_length` must be  > 0"

    @assert (0 <= max_infogain_ratio <= 1) "max_infogain_ratio must be in range [0,1], but $(max_infogain_ratio) encountered."

    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."

    searchmethod = safe_reconstruct(searchmethod, kwargs)

    info_dl = (;
        supporting_labels=y,
    )

    y, labels = y |> maptointeger
    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]       # Il rulebase effettivo
    while true

        # bestantecedent_coverage è un array di 0 e 1 con 1 negli indici i dove la regola trovata copre il sample xi (in unconveredX)
        bestantecedent = findbestantecedent(searchmethod, uncoveredX, uncoveredy, uncoveredw,
            #
            loss_function,
            max_infogain_ratio,
            default_alphabet,
            discretizedomain,
            significance_alpha,
            min_rule_coverage; 
            max_rule_length=max_rule_length,
            nlabels=length(labels)
        )

        istop(bestantecedent) && break

        rule = begin
            justcoveredy = uncoveredy[bestantecedent.covmask]
            justcoveredw = uncoveredw[bestantecedent.covmask]
            # indice della classe associata alla regola
            predlabel = SoleModels.bestguess(labels[justcoveredy], justcoveredw; suppress_parity_warning=suppress_parity_warning)

            info_cm = (;
                supporting_labels=[labels[x] for x in collect(justcoveredy)],       # array con i labels dei sample appena coperti dalla regola creata
                supporting_predictions=fill(predlabel, length(justcoveredy)),       # array dove per ogni sample appena coperto c'è il label che la nuova regola gli assegna 
            )
            consequent = ConstantModel(predlabel, info_cm)

            # info della struct regola appena trovata
            info_r = (;
                supporting_labels=[labels[x] for x in collect(uncoveredy)],
            )
            Rule(bestantecedent.formula, consequent, info_r)
        end

        push!(rulebase, rule)

        # indici dei samples non ancora coperti dalla nuova regola (e quindi da nessun'altra)
        uncovered_slice = (!).(bestantecedent.covmask)

        # da SoleData
        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]

        if !isnothing(max_rulebase_length) && length(rulebase) > (max_rulebase_length - 1)
            break
        end
    end
    prediction = SoleModels.bestguess(uncoveredy; suppress_parity_warning=suppress_parity_warning)
    prediction = labels[prediction]
    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncoveredy)],
        supporting_predictions=fill(prediction, length(uncoveredy)),
    )
    default_consequent = ConstantModel(prediction, info_cm)
    return DecisionList(rulebase, default_consequent, info_dl)
end


############################################################################################
################### SequentialCovering - IREP* ######################################
############################################################################################





function irepstar(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector{U} = default_weights(length(y));
    kwargs...
)::DecisionList where {U<:Real}
    # TODO: scrivere tutti i check sull'input
    @assert length(y) == ninstances(X) "The sizes of the training data X and of the labels y do not match. X has $num_instances instances, whilst y has length $(length(y))"
    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]
    

    # corresponding to the integer value in the targets in y
    y_int, labels = y |> maptointeger
    y_dist = counts(y_int)    # y_dist[i] is the number of times the label i occurs in y_int

    # indici ordinati delle classi in ordine crescente di copertura
    sorted_indices = sortperm(y_dist)

    # the actual list of rules that is obtained from the various calls 
    rules = Rule[]

    uncovered = TrainingState(X, y, w)

    # starting from the least common class index and going up to the most common, the last class
    # is used as the default consequent
    for class_idx ∈ sorted_indices[1:end-1]
        # Create the decision list with a call to IREP* on the data that still hasn't been classified
        label = labels[class_idx]

        num_positive_samples = count(x -> x == label, uncovered.y)       # number of samples with the label we're looking for in the dataset
        if num_positive_samples == 0
            continue end

        Base.@debug "Training binary classification IREP* model for class $label"
        
        class_decision_list = irepstar(
            uncovered.X, uncovered.y, label,
            uncovered.w;
            kwargs...
        )

        # rules learned for this class, each has its own antecedent, support and consequent
        class_rules = rulebase(class_decision_list)
        append!(rules, class_rules)
        

        # Extract the coverage mask for the decision list just created, and from that the indices just covered by the decision list for the label class_idx
        declist_prediction = apply(class_decision_list, uncovered.X)
        covered_indices = findall(x -> x == label, declist_prediction)        
        
        
        Base.@debug begin
            binary_acc = ModalDecisionLists.Metrics.binary_accuracy(uncovered.y, declist_prediction, label)
            "Binary accuracy for target class $label: $binary_acc"
        end
  

        # all the samples have been covered
        if length(covered_indices) == ninstances(uncovered.X)
            Base.@debug "All the samples have been covered, exiting the training loop"
            break end

        # Remove the covered samples from the uncovered slice of the dataset
        uncovered_slice = setdiff(1:ninstances(uncovered.X), covered_indices)
        uncovered = sliceinstances(uncovered, uncovered_slice; return_view = true)
    end

    # The most populated class in the dataset is predicted as default when no other previously discovered rule applies
    default_class_index = sorted_indices[end]
    default_class = labels[default_class_index]     # default prediction if no other rule applies

    Base.@debug "Resorting to default class $default_class if no other rule applies"

    info_cm = (;
        # supporting_labels=[labels[x] for x in collect(uncovered_original_y)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(default_class, length(y)),
    )

    default_consequent = ConstantModel(default_class, info_cm)
    
    info_dl = (;
        supporting_labels=y,
        supporting_weights=w
        # TODO: add supporting predictions?
    )
    
    return DecisionList(rules, default_consequent, info_dl)
end



function irepstar(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    poslabel::CLabel,
    w::AbstractVector{U} = default_weights(length(y));

    searchmethod::SearchMethod = BeamSearch(), 
    tdl_threshold::Int=64,
    split_ratio::Real=0.7, 
    loss_function::ModalDecisionLists.LossFunctions.AsymmetricLoss = ModalDecisionLists.LossFunctions.LaplaceAccuracy(),
    max_infogain_ratio::Union{Nothing,Real}=nothing,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    discretizedomain::Bool=false,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Integer=1, 
    max_rule_length::Union{Nothing,Integer}=nothing,
    max_rulebase_length::Union{Nothing,Integer}=nothing,

    rng::AbstractRNG = Random.default_rng(),
    suppress_parity_warning::Bool=false,

    kwargs...
)::DecisionList where {U<:Real}

    !isnothing(max_rulebase_length) && @assert max_rulebase_length > 0 "`max_rulebase_length` must be  > 0"

    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]
    !isnothing(max_infogain_ratio) && @assert (0 <= max_infogain_ratio <= 1) "Parameter `max_infogain_ratio` must be in range [0,1], but $(max_infogain_ratio) encountered."

    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter `max_rule_length` cannot be less" *
                                                               "than one. Please provide a valid value."

    @assert (0 < split_ratio < 1) "Parameter `split_ratio` must be in range (0,1)"
    @assert (min_rule_coverage > 0) "Parameter `min_rule_coverage` must be ≥ 1"

    # in Parameters.jl
    searchmethod = safe_reconstruct(searchmethod, kwargs)

    info_dl = (;
        supporting_labels=y,
    )

    y, labels = y |> maptointeger
    poslabel_idx = findfirst(x -> x == poslabel, labels) # indice in labels della classe positiva)

    @assert !isnothing(poslabel_idx) "The dataset provided must contain at least one positive sample!"

    @assert length(y) == ninstances(X) "The sizes of the training data X and of the labels y do not match. X has $num_instances instances, whilst y has length $(length(y))"

    uncovered_original_y = y
    y = UInt32.(y .== poslabel_idx) # convert y to an array of {0,1}, with 1 being the target class and 0 being anything else

    # samples yet to be covered by any Rule in the RuleSet
    uncovered = TrainingState(X, y, w, uncovered_original_y)

    rulebase_sat_mask = falses(ninstances(X))   # sat mask della rulebase su uncoveredX
    data_curr_ruleset_desc_length = Inf
    dataset_num_selectors = get_num_independent_selectors(X, y, discretizedomain)
    
    curr_TDL = get_initial_dataset_bits(y)
    curr_min_TDL = curr_TDL
    
    rulebase = Rule[]
    while true

        if !isnothing(max_rulebase_length) && length(rulebase) >= max_rulebase_length
            @debug "The IREP* loop was stopped because the specified maximum amount of rules ($max_rulebase_length) has been reached"
            break
        end

        split = split_instances(uncovered.X, uncovered.y, uncovered.w, split_ratio, rng)
        split === nothing && break

        num_uncovered_pos = count(label -> label == 1, uncovered.y)    # total number of uncovered positive samples

        if num_uncovered_pos < min_rule_coverage
            Base.@debug "Training converged because the number of positive samples remaining is lower than min_rule_coverage = $min_rule_coverage"
            break end

        bestantecedent = findbestantecedent(searchmethod,
            growth_X(split), growth_y(split), growth_w(split),
            #
            loss_function,
            max_infogain_ratio,
            default_alphabet,
            discretizedomain,
            significance_alpha,
            min_rule_coverage; 
            max_rule_length=max_rule_length,
            nlabels=2,
            target_class=1
        )

        Base.@debug begin
            # 1. Total distribution in the current split
            tp_potential = count(==(1), split.gr.y)
            fp_potential = count(!=(1), split.gr.y)

            # 2. Coverage counts (True Positives and False Positives)
            covered_labels = @views split.gr.y[bestantecedent.covmask]
            
            rule_tp = count(==(1), covered_labels)
            rule_fp = count(!=(1), covered_labels) 
            
            # Total samples covered by the rule
            total_covered = rule_tp + rule_fp

            """
            Antecedent developed: $bestantecedent
            Split Totals: (Neg: $fp_potential, Pos: $tp_potential)
            Rule Performance:
            - True Positives (TP): $rule_tp
            - False Positives (FP): $rule_fp
            - Total Covered: $total_covered
            """
        end

        istop(bestantecedent) && break

        # PRUNING
        bestantecedent, bestantecedent_prune_cov = pruneantecedent(bestantecedent, prune_X(split), prune_y(split), prune_w(split))

        # Create the new Rule as an instance of "Rule" from SoleModels
        coverage_indices = compute_global_coverage(bestantecedent, split, bestantecedent_prune_cov)
        rule = build_rule(bestantecedent, uncovered.original_y, poslabel, coverage_indices, labels)


        # Calculate the description length of the new Rule
        rule_desc_length = _r_theory_bits(rule, dataset_num_selectors)

        push!(rulebase, rule)

        data_new_ruleset_desc_length, rulebase_sat_mask = rs_dataset_bits(X, y, rule, rulebase_sat_mask)


        Base.@debug "Description length of the new Rule: $rule_desc_length"
        Base.@debug "Description length of the dataset given with the addition of the new rule to the ruleset: $data_new_ruleset_desc_length"


        # ΔTDL = ΔTDL(Ruleset) + ΔTDL(Dataset | Ruleset), dove ΔTDL(Ruleset) = TDL(Ruleset + Rule_i) - TDL(Ruleset) = TDL(Rule_i), with TDL being the Total Description Length
        # In other words, since every iteration adds a single Rule to the RuleSet, the description length of the ruleset increases by the description length of the rule
        # this is why ΔTDL_ruleset is just rule_desc_length
        ΔTDL_data_given_ruleset = data_new_ruleset_desc_length - data_curr_ruleset_desc_length
        ΔTDL_ruleset = rule_desc_length
        ΔTDL = ΔTDL_ruleset + ΔTDL_data_given_ruleset
        curr_TDL += ΔTDL

        Base.@debug begin
            printed_diff = isinf(data_curr_ruleset_desc_length) ? rule_desc_length + data_new_ruleset_desc_length : ΔTDL
            "Difference in Total Description Length: $printed_diff"
        end

        # Stopping condition + curr_min_TDL update condition
        if curr_TDL > curr_min_TDL + tdl_threshold
            pop!(rulebase)
            break
        elseif curr_TDL < curr_min_TDL
            curr_min_TDL = curr_TDL
        end


        data_curr_ruleset_desc_length = data_new_ruleset_desc_length

        # Calculate the indices that remain uncovered after the new rule has been added to the RuleSet
        uncovered_slice = setdiff(1:ninstances(uncovered.X), coverage_indices)
        
        # Stop if the entire dataset has been covered, otherwise slicedataset would throw an error
        if length(uncovered_slice) == 0
            break end
        
        
        Base.@debug "Number of uncovered samples remaining: $(length(uncovered_slice))"


        uncovered = sliceinstances(uncovered, uncovered_slice; return_view = true)
    end

    default_prediction = "other"    # default prediction if no other Rule applies

    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncovered.original_y)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(default_prediction, length(uncovered.original_y)),
    )
    default_consequent = ConstantModel(default_prediction, info_cm)
    return DecisionList(rulebase, default_consequent, info_dl)
end



"""
    compute_global_coverage(antecedent::LeftmostConjunctiveForm, split, bestantecedent_prune_cov)

Compute the global coverage of an antecedent rule across both grow and prune datasets.

This function evaluates how many instances a given antecedent covers by checking it against
both the grow and prune subsets of the data. It maps local indices (indices within each subset)
back to global indices using the provided index mappings.

# Arguments
- `antecedent::LeftmostConjunctiveForm`: The antecedent rule to evaluate coverage for.
- `split`: A data structure containing grow and prune dataset splits with their corresponding
  feature matrices (`gr.X`, `pr.X`) and global index mappings (`gr_inds`, `pr_inds`).
- `bestantecedent_prune_cov`: Either a boolean mask indicating which instances in the prune
  set are covered by a previously computed best antecedent, or `nothing` if no cached mask
  is available. If provided, this is used instead of recomputing the mask.

# Returns
A vector of global indices representing all instances covered by the antecedent across both
grow and prune datasets.
"""
function compute_global_coverage(
    antecedent::LeftmostConjunctiveForm,
    split::DataSplit,
    bestantecedent_prune_cov::Union{Nothing, Vector{<:Bool}, BitVector}
)
    growX = growth_X(split)
    pruneX = prune_X(split)

    grow_mask = check(antecedent, growX)
    grow_cov_local = findall(grow_mask)
    grow_cov_global = grow_indices(split)[grow_cov_local]

    if bestantecedent_prune_cov === nothing
        prune_mask = check(antecedent, pruneX)
    else
        prune_mask = bestantecedent_prune_cov
    end
    prune_cov_local = findall(prune_mask)
    prune_cov_global = prune_indices(split)[prune_cov_local]

    return vcat(grow_cov_global, prune_cov_global)
end


"""
    build_rule(antecedent, uncovered_original_y, poslabel, coverage_indices, labels)

Create a `Rule` instance using the provided `antecedent`. 

The function constructs a `ConstantModel` as the consequent (prediction) based on 
the provided `poslabel` and attaches metadata regarding the samples covered 
by the rule.

# Arguments
- `antecedent`: The rule's antecedent/condition.
- `uncovered_original_y`: Vector of class integers for the current dataset.
- `poslabel`: The label to be assigned as the prediction.
- `coverage_indices`: Indices of the samples covered by the rule.
- `labels`: The original mapping of class integers to `CLabel` objects.
"""
function build_rule(
    antecedent::LeftmostConjunctiveForm, 
    uncovered_original_y::AbstractVector, 
    poslabel::CLabel, 
    coverage_indices::AbstractVector{<:Integer}, 
    labels::AbstractVector{<:CLabel}
)
    justcoveredy = uncovered_original_y[coverage_indices]
    predlabel = poslabel

    info_cm = (;
        supporting_labels=[labels[x] for x in collect(justcoveredy)],
        supporting_predictions=fill(predlabel, length(justcoveredy)),
    )
    consequent = ConstantModel(predlabel, info_cm)

    info_r = (;
        supporting_labels=[labels[x] for x in collect(uncovered_original_y)],
    )

    return Rule(antecedent, consequent, info_r)
end





"""
    generate_pruned_rules(rule::LeftmostConjunctiveForm)

Generates all the pruned versions of the rule `rule`, obtained from all the prefixes of its antecedent,
and returns them as a list of Formulas in order of decreasing length (startin from the complete rule down to its first atom).

This is used in the pruning phase in RIPPER
"""

# TODO: @Nicola: specificare un ulteriore parametro per il pruning: 
# Esistono metodi alternativi oper il pruning invece che rimuovere in maniera monotona l'ultima condizione ? 
# Questo andrebbe fatto in una propria struct effettiva, magari con dispatching sulle varie pruning strategies.
# Bisogna prima identificare qualche metodo che si vuole implementare poi si pensa a tutta la struttura effettiva
function generate_pruned_formulas(ant::Antecedent)
    _range = nconds(ant):-1:1
    return [LeftmostConjunctiveForm(conds(ant)[1:i])
            for i in _range
    ]
end

function pruneantecedent(
    antecedent::Antecedent,
    X::AbstractLogiset,
    y::AbstractVector{UInt32},
    w::AbstractVector{U} = default_weights(length(y))
) where {U<:Real}
    target_class = 1

    # 1. Build the positive/negative masks with respect to the target class
    posmask = y .== target_class
    negmask = .!posmask

    # 2. Initialization of the best antecedent (best rule) 
    _best_formula = antecedent.formula
    _best_covmask = nothing
    _best_score = -Inf              # this makes sure that at least one formula will be selected as _best_formula in the loop
    

    # 3. Evaluate all possible pruned versions of the formula, including the original formula itself
    for pformula in generate_pruned_formulas(antecedent)

        p_covmask = check(pformula, X)

        p = sum(w[posmask .& p_covmask]) # Sum of True positives weight values
        n = sum(w[negmask .& p_covmask]) # Sum of False positives weight values


        # v* (IREP* pruning criterion)
        score = (p + n != 0) ? (p - n) / (p + n) : -1.0     # -1 is the lowest possible value of (p-n)/(p+n)

        if score > _best_score
            _best_formula = pformula
            _best_covmask = p_covmask
            _best_score = score
        end
    end

    return _best_formula, _best_covmask
end


function get_num_independent_selectors(
    X::AbstractLogiset, 
    y::AbstractVector{<:UInt32}, 
    discretizedomain::Bool=false
)::Int
    alph = alphabet(X;
        discretizedomain=discretizedomain,
        y=y
    )

    independent_conds = alphabet2conditions(AtomGenerator(), alph, X)
    return length(independent_conds)
end

"""
    function _r_theory_bits(rule::Rule, n_possible_conds::Int)::Int

    Returns the TDL (Total Description Length) of a Rule
"""
function _r_theory_bits(rule::Rule, n::Int)
    # conds = unaryconditions_noneq(alph, X)        # @Nicola va richiamato? su wittgenstein sembra sia fissato ma mi puzza come cosa
    # n_old = length(conds) 

    k = 1 + nconnectives(rule.antecedent)       # nconnectives is defined on any type <:Formula
    pr = k / n

    S = k * log2(1 / pr) + (n - k) * log2(1 / (1 - pr))
    K = log2(k)
    desc_length = (S + K) * 0.5

    return max(desc_length, 1)
end


""" returns an approximation of ln(n!) using Stirling's approximation for numerical stability and optimization """
log2_factorial(n::Integer)::Real = (n == 0) ? 0 : max(0, 0.5 * (1 + log2(π * n)) + n * log2(n / ℯ) + 0.1201753 / n)     # 0.1201753 / n is just a term that minimizes the approximation whilst reducing error


""" returns an approximation of ln( n choose k ) using log2_factorial for numerical stability and optimization  """
log2binomial(n::Integer, k::Integer)::Real = (k == 0) ? 0 : log2_factorial(n) - log2_factorial(k) - log2_factorial(n - k)


""" In a particular binary classification problem, this function returns the number of bits to describe the dataset (X,y) 
given a ruleset with the satisfaction mask "ruleset_satmask" """
function rs_dataset_bits(
    y::AbstractVector{<:UInt32},
    ruleset_satmask::BitVector
)
    n_samples = length(y)
    num_pos = count(label -> label == 1, y)
    p = sum(ruleset_satmask)                   # number of samples covered by the ruleset

    ruleset_covered_labels = y[ruleset_satmask]
    tp = count(==(1), ruleset_covered_labels)
    fp = count(!=(1), ruleset_covered_labels)

    fn = num_pos - tp       # false negatives

    desc_length = log2binomial(p, fp) + log2binomial(n_samples - p, fn)
    return desc_length
end



""" In a particular binary classification problem, this function returns the number of bits to describe the dataset (X,y) 
given the previous satisfaction/coverage mask 'prev_ruleset_satmask' of the ruleset, and a new rule added to the ruleset """
function rs_dataset_bits(
    X::AbstractLogiset, 
    y::AbstractVector{<:UInt32},
    rule::Rule,
    prev_ruleset_satmask::BitVector
)
    rule_sat_mask = check(rule.antecedent, X)       # check which samples are covered by the new rule
    ruleset_sat_mask = prev_ruleset_satmask .| rule_sat_mask    # update the sat mask of the whole ruleset by adding samples covered by the new rule

    return rs_dataset_bits(y, ruleset_sat_mask), ruleset_sat_mask
end

""" In a binary classification problem, this function returns the number of bits required to describe a dataset with labels y, such that
yᵢ ∈ {0,1}, given the initial default rule ⊥ that assigns the negative class 0 to every sample."""
function get_initial_dataset_bits(
    y::AbstractVector{<:UInt32}
)
    # the default classification "other" is a negative classification, which assigns 0 (false) to every sample.
    # The number of true positives and false positives will be 0, the number of false negatives will be the number of true samples
    # the number of true negatives will be the number of negative samples
    # tp = fp = 0; tn = n, fn = p 
    n_samples = length(y)
    n_positive_samples = count(x -> x == 1, y)
    
    return log2binomial(n_samples, n_positive_samples)
end






############################################################################################
################### RIPPERk - Binary Classification #######################################
############################################################################################


function ripperk(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector{U} = default_weights(length(y));
    kwargs...
)::DecisionList where {U<:Real}
    # TODO: scrivere tutti i check sull'input
    @assert length(y) == ninstances(X) "The sizes of the training data X and of the labels y do not match. X has $num_instances instances, whilst y has length $(length(y))"
    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]
    

    # corresponding to the integer value in the targets in y
    y_int, labels = y |> maptointeger
    y_dist = counts(y_int)    # y_dist[i] is the number of times the label i occurs in y_int

    # indici ordinati delle classi in ordine crescente di copertura
    sorted_indices = sortperm(y_dist)

    # the actual list of rules that is obtained from the various calls 
    rules = Rule[]

    uncovered = TrainingState(X, y, w)

    # starting from the least common class index and going up to the most common, the last class
    # is used as the default consequent
    for class_idx ∈ sorted_indices[1:end-1]
        # Create the decision list with a call to IREP* on the data that still hasn't been classified
        label = labels[class_idx]

        num_positive_samples = count(x -> x == label, uncovered.y)       # number of samples with the label we're looking for in the dataset
        if num_positive_samples == 0
            continue end

        Base.@debug "Training binary classification IREP* model for class $label"
        
        class_decision_list = ripperk(
            uncovered.X, uncovered.y, label,
            uncovered.w;
            kwargs...
        )

        # rules learned for this class, each has its own antecedent, support and consequent
        class_rules = rulebase(class_decision_list)
        append!(rules, class_rules)
        

        # Extract the coverage mask for the decision list just created, and from that the indices just covered by the decision list for the label class_idx
        declist_prediction = apply(class_decision_list, uncovered.X)
        covered_indices = findall(x -> x == label, declist_prediction)        
        
        
        Base.@debug begin
            binary_acc = ModalDecisionLists.Metrics.binary_accuracy(uncovered.y, declist_prediction, label)
            "Binary accuracy for target class $label: $binary_acc"
        end
  

        # all the samples have been covered
        if length(covered_indices) == ninstances(uncovered.X)
            Base.@debug "All the samples have been covered, exiting the training loop"
            break end

        # Remove the covered samples from the uncovered slice of the dataset
        uncovered_slice = setdiff(1:ninstances(uncovered.X), covered_indices)
        uncovered = sliceinstances(uncovered, uncovered_slice; return_view = true)
    end

    # The most populated class in the dataset is predicted as default when no other previously discovered rule applies
    default_class_index = sorted_indices[end]
    default_class = labels[default_class_index]     # default prediction if no other rule applies

    Base.@debug "Resorting to default class $default_class if no other rule applies"

    info_cm = (;
        # supporting_labels=[labels[x] for x in collect(uncovered_original_y)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(default_class, length(y)),
    )

    default_consequent = ConstantModel(default_class, info_cm)
    
    info_dl = (;
        supporting_labels=y,
        supporting_weights=w
        # TODO: add supporting predictions?
    )
    
    return DecisionList(rules, default_consequent, info_dl)
end




function ripperk(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    poslabel::CLabel,
    w::AbstractVector{U} = default_weights(length(y));
    
    max_k::Integer = 1,

    searchmethod::SearchMethod = BeamSearch(), 
    tdl_threshold::Int=64,
    split_ratio::Real=0.7, 
    loss_function::ModalDecisionLists.LossFunctions.AsymmetricLoss = ModalDecisionLists.LossFunctions.LaplaceAccuracy(),
    max_infogain_ratio::Union{Nothing,Real}=nothing,
    default_alphabet::Union{Nothing,AbstractAlphabet}=nothing,
    discretizedomain::Bool=false,
    significance_alpha::Union{Real,Nothing}=0.0,
    min_rule_coverage::Integer=1, 
    max_rule_length::Union{Nothing,Integer}=nothing,
    max_rulebase_length::Union{Nothing,Integer}=nothing,

    rng::AbstractRNG = Random.default_rng(),
    suppress_parity_warning::Bool=false,


    kwargs...
)::DecisionList where {U<:Real}

    num_samples = ninstances(X)

    @assert max_k > 0 "Parameter `max_k` must be > 0"
    @assert length(y) == num_samples "The sizes of the training data X and of the labels y do not match."

    Base.@debug "Starting RIPPERk optimization with max_k=$max_k iterations"

    searchmethod = safe_reconstruct(searchmethod, kwargs)

    info_dl = (;
        supporting_labels=y,
    )


    uncovered_original_y_labels = y
    y, labels = y |> maptointeger
    poslabel_idx = findfirst(x -> x == poslabel, labels)

    uncovered_original_y = y
    y = UInt32.(y .== poslabel_idx)  # convert y to an array of {0,1}, with 1 being the target class and 0 being anything else

    uncovered = TrainingState(X, y, w, uncovered_original_y, uncovered_original_y_labels)

    # Create the initial ruleset, keeping it only as a vector of rules
    Base.@debug "Creating initial ruleset through call to IREP*"
    curr_ruleset = irepstar(uncovered.X, uncovered.original_y_labels, poslabel, uncovered.w; 
                            searchmethod = searchmethod,
                            tdl_threshold = tdl_threshold,
                            split_ratio = split_ratio, 
                            loss_function = loss_function,
                            max_infogain_ratio = max_infogain_ratio,
                            default_alphabet = default_alphabet,
                            discretizedomain = discretizedomain,
                            significance_alpha = significance_alpha,
                            min_rule_coverage = min_rule_coverage,
                            max_rule_length = max_rule_length,
                            max_rulebase_length = max_rulebase_length,
                            rng = rng, 
                            suppress_parity_warning = suppress_parity_warning,
                            kwargs...)

        
    curr_ruleset = rulebase(curr_ruleset)

    
    # This cannot possibly be inside the loop, otherwise the description length of a rule would change based on the ripper_iteration, it just doesn't make sense
    num_selectors = get_num_independent_selectors(X, y, discretizedomain)       
    
    for ripper_iteration = 1 : max_k
        ruleset_masks = _precalculate_rules_satmasks(uncovered.X, curr_ruleset)
        
        # Calculate initial TDL 
        curr_tdl = _calculate_TDL(uncovered.y, curr_ruleset, num_selectors, ruleset_masks)
        optimized_ruleset_satmask = falses( ninstances(uncovered.X) )                # whilst we optimize the rules, we also calculate which samples are covered by the new ruleset
        args = (loss_function, max_infogain_ratio, default_alphabet, discretizedomain, significance_alpha, min_rule_coverage)       # Findbestantecedent args
        
        optimized_ruleset_satmask = _optimize_ruleset!(
            ruleset_masks, curr_ruleset, uncovered,
            labels, poslabel, args, curr_tdl, searchmethod, num_selectors, 
            split_ratio, rng, max_rule_length 
        )


        # Calculate indices covered and not covered by the ruleset
        covered_indices = findall(x -> x == 1, optimized_ruleset_satmask)
        uncovered_slice = setdiff(1:ninstances(uncovered.X), covered_indices)
        
        Base.@debug "Number of uncovered samples remaining: $(length(uncovered_slice))"
        
        # Only keep the remaining uncovered samples
        if length(uncovered_slice) == 0
            break end

        uncovered = sliceinstances(uncovered, uncovered_slice; return_view = true)

        # Check for stopping condition if no new rule can be made with such few samples. This also handles the case where no positive samples are remaining
        num_pos_samples_remaining = count(x -> x == 1, uncovered.y)
        if num_pos_samples_remaining < min_rule_coverage
            Base.@debug "RIPPER training stopped after iteration $ripper_iteration because the number of positive samples remaining was smaller than min_rule_coverage"
            break
        end

        # Call IREP* again to get the residual ruleset
        residual_ruleset = irepstar(uncovered.X, uncovered.original_y_labels, poslabel, uncovered.w; 
                            searchmethod = searchmethod,
                            tdl_threshold = tdl_threshold,
                            split_ratio = split_ratio, 
                            loss_function = loss_function,
                            max_infogain_ratio = max_infogain_ratio,
                            default_alphabet = default_alphabet,
                            discretizedomain = discretizedomain,
                            significance_alpha = significance_alpha,
                            min_rule_coverage = min_rule_coverage,
                            max_rule_length = max_rule_length,
                            max_rulebase_length = max_rulebase_length,
                            rng = rng, 
                            suppress_parity_warning = suppress_parity_warning,
                            kwargs...)

        # Append the residual ruleset to 
        residual_ruleset = rulebase(residual_ruleset)
        append!(curr_ruleset, residual_ruleset)
    end

    default_prediction = "other"    # default prediction if no other Rule applies

    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncovered.original_y)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(default_prediction, length(uncovered.original_y)),
    )
    default_consequent = ConstantModel(default_prediction, info_cm)

    Base.@debug "RIPPERk optimization complete"
    return DecisionList(curr_ruleset, default_consequent, info_dl)
end




function _optimize_ruleset!(
    ruleset_masks::BitMatrix,
    curr_ruleset::AbstractVector{<:Rule},
    uncovered::TrainingState,
    labels::AbstractVector{<:CLabel},
    poslabel::CLabel,
    args::Tuple,
    curr_tdl::Real,
    
    searchmethod::SearchMethod,
    num_selectors::Integer,
    split_ratio::Real,
    rng::AbstractRNG,
    max_rule_length::Union{Nothing, Integer}
)

    optimized_ruleset_satmask = falses( ninstances(uncovered.X) )                # whilst we optimize the rules, we also calculate which samples are covered by the new ruleset


    for (i, rule) ∈ enumerate(curr_ruleset)
        original_rule_satmask = ruleset_masks[:, i]         # cache the satmask of the current rule
        original_rule_covered_indices = findall(x -> x == 1, original_rule_satmask)
        default_dataset_satmask = merge_ruleset_satmasks(ruleset_masks, i)      # satmask of the dataset if 'rule' was not a part of it

        # generate a Grow/Prune split of the data to be used to grow and prune other variants of the rule
        split = split_instances(uncovered.X, uncovered.y, uncovered.w, split_ratio, rng)
        split === nothing && break
        

        # Consider newly grown rule as a variant to rule
        # grow a new rule and prune it, 
        # args = (loss_function, max_infogain_ratio, default_alphabet, discretizedomain, significance_alpha, min_rule_coverage)       # Findbestantecedent args
        rule_grown, rule_grown_covered_indices = _grow_and_prune_rule(
            searchmethod, split, uncovered.original_y, poslabel, 
            labels, default_dataset_satmask, args; 
            nlabels = 2, max_rule_length = max_rule_length,
            target_class = 1
        )
        if rule_grown === nothing
            rule_grown = rule
            rule_grown_covered_indices = original_rule_covered_indices
        end

        # substitute the new rule's sat mask in place of the i-th rule, and use the resulting sat matrix to calculate the TDL of the entire ruleset
        ruleset_masks[:, i] .= false                       
        ruleset_masks[rule_grown_covered_indices, i] .= true
        curr_ruleset[i] = rule_grown
        rule_grown_tdl = _calculate_TDL(uncovered.y, curr_ruleset, num_selectors, ruleset_masks)



        # refine a new rule starting from 'rule' and prune it, do the same as before
        rule_revised, rule_revised_covered_indices = _revise_and_prune_rule(
            searchmethod, split, uncovered.original_y, poslabel, 
            labels, default_dataset_satmask, 
            rule, original_rule_satmask, args; 
            nlabels = 2, max_rule_length = max_rule_length,
            target_class = 1
        )
        if rule_revised === nothing
            rule_revised = rule
            rule_revised_covered_indices = original_rule_covered_indices
        end

        ruleset_masks[:, i] .= false                       
        ruleset_masks[rule_revised_covered_indices, i] .= true
        curr_ruleset[i] = rule_revised
        rule_revised_tdl = _calculate_TDL(uncovered.y, curr_ruleset, num_selectors, ruleset_masks)

        # Select best rule amongst the three
        competing_TDLs = (curr_tdl, rule_grown_tdl, rule_revised_tdl)
        competing_rules = (rule, rule_grown, rule_revised)
        competing_rules_coverage_indices = (original_rule_covered_indices, rule_grown_covered_indices, rule_revised_covered_indices)

        # Extract best rule
        best_tdl_idx = argmin(competing_TDLs)
        best_rule = competing_rules[best_tdl_idx]

        # Replace current rule with the best one
        curr_ruleset[i] = best_rule 
        curr_tdl = competing_TDLs[best_tdl_idx]
        chosen_rule_coverage_indices = competing_rules_coverage_indices[best_tdl_idx]
        ruleset_masks[:, i] .= false
        ruleset_masks[chosen_rule_coverage_indices, i] .= true

        
        Base.@debug begin
            """=========== Rule optimization #$i ===========
            Total description lengths for competing rules (original, grown, revised): $(round.(competing_TDLs, digits=3))
            Best rule: $best_rule
            Best rule index: $best_tdl_idx
            original rule's covered indices: $original_rule_covered_indices"""
        end

        optimized_ruleset_satmask[chosen_rule_coverage_indices] .= true

    end     # ruleset optimization completed 

    return optimized_ruleset_satmask
end




"""
    _precalculate_rules_satmasks(X::AbstractLogiset, rules::AbstractVector{<:Rule})

Precompute satisfaction masks for all rules across all samples.

Creates a BitMatrix where each column contains the satisfaction mask for one rule,
representing which samples are covered by that rule. This precomputation can improve
performance when evaluating multiple rules repeatedly.
Since bit-wise operations between different masks are common, it's more efficient
to store a single rule's coverage mask in a column rather than in a row, so that
two rules' coverage masks are adjacent in memory bit-by-bit.

# Arguments
- `X::AbstractLogiset`: The dataset to evaluate rules on.
- `rules::AbstractVector{<:Rule}`: A vector of rules to precompute masks for.

# Returns
A `BitMatrix` of size `(num_samples × num_rules)` where each column `i` contains
the satisfaction mask for rule `i`.
"""
function _precalculate_rules_satmasks(
    X::AbstractLogiset,
    rules::AbstractVector{<:Rule},
)::BitMatrix
    # A matrix of size (num_samples x num_rules). The i-th column is the i-th rule's sat mask
    n_rules = length(rules)
    num_samples = ninstances(X)
    ruleset_masks = BitMatrix(falses(num_samples, n_rules))
    
    for rule_idx = 1 : n_rules
        rule = rules[rule_idx]
    
        rule_satmask = checkantecedent(rule, X)
        ruleset_masks[:, rule_idx] = rule_satmask
    end

    return ruleset_masks
end


"""
    _calculate_ruleset_length(X, y, rules)

Calculate the total description length of a ruleset and of some data given that ruleset.
"""
function _calculate_TDL(
    y::AbstractVector{<:UInt32},
    rules::AbstractVector{<:Rule},
    num_possible_selectors::Int,
    ruleset_masks::BitMatrix
)::Real
    # Calculate Description length of the ruleset itself, ignoring data (TDL(Ruleset)), we also use the loop to calculate the satmask of the ruleset
    num_samples = length(y)
    ruleset_satmask = falses(num_samples)

    ruleset_dl = 0.0
    for (i, rule) ∈ enumerate(rules)
        ruleset_dl += _r_theory_bits(rule, num_possible_selectors)
        
        rule_satmask = @view ruleset_masks[:, i]
        ruleset_satmask = ruleset_satmask .| rule_satmask
    end

    # Calculate description length of the data, given the ruleset
    data_dl_given_ruleset = rs_dataset_bits(y, ruleset_satmask)

    return ruleset_dl + data_dl_given_ruleset
end


""" Given a dataset (X,y), a fixed ruleset with a coverage mask 'ruleset_mask' and an antecedent 'ant', this function prunes 'ant' using
reduced error pruning over a "joint coverage hypothesis" given by (ruleset(x) v ant(x)). In other words, this prunes ant
in order to maximize the accuracy if ant were to be added to the ruleset.  """
function reduced_error_prune_rule(
    X::AbstractLogiset,
    y::AbstractVector{<:UInt32},
    w::AbstractVector,
    ruleset_mask::BitVector,
    ant::Antecedent
)

    # Build positive/negative masks with respect to the target class
    target_mask = (y .== 1)

    _best_formula = ant.formula
    _best_covmask = nothing
    _best_errors_sum = Inf

    # Evaluate all pruned versions of the formula, including the original formula itself
    for pformula ∈ generate_pruned_formulas(ant)
        p_covmask = check(pformula, X)
        total_ruleset_covmask = ruleset_mask .| p_covmask           # satmask of the ruleset if we were to add the rule with antecedent formula "pformula" to the ruleset

        errors_mask = (total_ruleset_covmask .!= target_mask)       # 1 where the formula's coverage and the actual {0,1} label differ
        weighted_errors_sum = sum(errors_mask .& w)                 # extract corresponding weights and sum

        # TODO: would "weighted_errors_sum = sum(w[ total_ruleset_covmask .!= target_mask ]) be faster? It might avoid summing tons of zeros
        
        if weighted_errors_sum < _best_errors_sum
            _best_formula = pformula
            _best_covmask = p_covmask
            _best_errors_sum = weighted_errors_sum
        end

    end

    return _best_formula, _best_covmask

end


""" Grows a rule, and then prunes it (using Reduced Error Pruning), in order to minimize the error of an entire ruleset over
a set of pruning samples (obtained from the split.pr attribute).
# Returns
The grown rule and the indices of the samples it covers.  """
function _grow_and_prune_rule(
    sm::SearchMethod,
    split::DataSplit,
    uncovered_original_y::AbstractVector,
    poslabel::CLabel,
    labels::AbstractVector{<:CLabel},
    default_dataset_satmask::BitVector,
    args;
    kwargs...
)
    # Growing
    bestantecedent = findbestantecedent(sm, growth_X(split), growth_y(split), growth_w(split), args...; kwargs...)
    istop(bestantecedent) && return nothing, nothing

    # PRUNING
    rule, coverage_indices = _prune_rule_over_dataset(split, default_dataset_satmask, bestantecedent, uncovered_original_y, poslabel, labels)
    return rule, coverage_indices
end

""" Revises a rule 'starting_rule' with coverage mask 'starting_rule_satmask', and then prunes it (using Reduced Error Pruning), in order to minimize the error of an entire ruleset over
a set of pruning samples (obtained from the split.pr attribute).
# Returns
The revised rule and the indices of the samples it covers. """
function _revise_and_prune_rule(
    sm::SearchMethod,
    split::DataSplit,
    uncovered_original_y::AbstractVector,
    poslabel::CLabel,
    labels::AbstractVector{<:CLabel},
    default_dataset_satmask::BitVector,

    starting_rule::Rule,
    starting_rule_satmask::BitVector,
    
    args;
    kwargs...
)
    # extract the coverage mask of the rule over the data from which the rule is to be grown
    antecedent_mask_over_grow_data = starting_rule_satmask[grow_indices(split)]

    rule_ant = Antecedent(starting_rule.antecedent, antecedent_mask_over_grow_data)

    revised_antecedent = findbestantecedent(sm, growth_X(split), growth_y(split), growth_w(split), args...; starting_antecedent = rule_ant, kwargs...)
    istop(revised_antecedent) && return nothing, nothing

    # PRUNING
    rule, coverage_indices = _prune_rule_over_dataset(split, default_dataset_satmask, revised_antecedent, uncovered_original_y, poslabel, labels)
    return rule, coverage_indices
end


""" Given an antecedent, a split, and a coverage mask of a ruleset (without the antecedent), this function uses the data from the pruning section of the split
to prune antecedent (through Reduced Error Pruning), in order to minimize the error of the entire ruleset over the pruning data """
function _prune_rule_over_dataset(
    split::DataSplit,
    default_dataset_satmask::BitVector,
    antecedent::Antecedent,
    uncovered_original_y::AbstractVector,

    poslabel::CLabel,
    labels::AbstractVector{<:CLabel}
)
    pruning_default_dataset_satmask = default_dataset_satmask[prune_indices(split)]
    pruned_ant, pruned_ant_covmask = reduced_error_prune_rule(prune_X(split), prune_y(split), prune_w(split), pruning_default_dataset_satmask, antecedent)
    
    coverage_indices = compute_global_coverage(pruned_ant, split, pruned_ant_covmask)
    
    # Transform Antecedent -> Rule
    rule = build_rule(pruned_ant, uncovered_original_y, poslabel, coverage_indices, labels)

    return rule, coverage_indices
end


""" Given the BitMatrix where each column is a satmask over a dataset for a certain rule, this function computes 
the total satmask for entire ruleset over the same dataset, excluding the result from excluded_rule_idx (if it's not nothing)"""
function merge_ruleset_satmasks(
    ruleset_satmasks::BitMatrix,
    excluded_rule_idx::Union{Nothing, Integer}
)
    num_samples = size(ruleset_satmasks, 1)     # num_samples = num_rows(ruleset_satmasks)
    num_rules = size(ruleset_satmasks, 2)        # num_rules = num_cols(ruleset_satmasks)
    ruleset_satmask::BitVector = falses(num_samples)

    for i = 1 : num_rules
        if !isnothing(excluded_rule_idx) && i == excluded_rule_idx
            continue end

        rule_satmask = @view ruleset_satmasks[:, i]
        ruleset_satmask = ruleset_satmask .| rule_satmask
    end

    return ruleset_satmask
end