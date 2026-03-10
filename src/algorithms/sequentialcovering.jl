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
using Random

############################################################################################
################### SequentialCovering - DecisionList ######################################
############################################################################################
# * `unorderedstrategy::Bool`: TODO @Edo explain
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

    @assert (0 <= max_infogain_ratio <= 1) "max_infogain_ratio must be in range [0,1], but $(maxpurity_gamma) encountered."

    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                               "than one. Please provide a valid value."

    searchmethod = reconstruct(searchmethod, kwargs)

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
            min_rule_coverage; max_rule_length=max_rule_length,
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
################### SequentialCovering - RIPPER ######################################
############################################################################################


function irepstar(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector{U} = default_weights(length(y));
    kwargs...
)::DecisionList where {U<:Real}
    # TODO: scrivere tutti i check sull'input
    
    # corresponding to the integer value in the targets in y
    y_int, labels = y |> maptointeger
    y_dist = counts(y_int)    # y_dist[i] is the number of times the label i occurs in y_int

    # indici ordinati delle classi in ordine crescente di copertura
    sorted_indices = sortperm(y_dist)

    # the actual list of rules that is obtained from the various calls 
    rules = Rule[]

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    # starting from the least common class index and going up to the most common, the last class
    # is used as the default consequent
    for class_idx ∈ sorted_indices[1:end-1]
        # Create the decision list with a call to IREP* on the data that still hasn't been classified
        label = labels[class_idx]

        num_positive_samples = count(x -> x == label, uncoveredy)       # number of samples with the label we're looking for in the dataset
        if num_positive_samples == 0
            continue end

        Base.@debug "Training binary classification IREP* model for class $label"
        
        class_decision_list = irepstar(
            uncoveredX, uncoveredy, label,
            uncoveredw;
            kwargs...
        )

        # rules learned for this class, each has its own antecedent, support and consequent
        class_rules = rulebase(class_decision_list)
        append!(rules, class_rules)
        

        # Extract the coverage mask for the decision list just created, and from that the indices just covered by the decision list for the label class_idx
        declist_prediction = apply(class_decision_list, uncoveredX)
        covered_indices = findall(x -> x == label, declist_prediction)        
        
        
        Base.@debug begin
            binary_acc = ModalDecisionLists.Metrics.binary_accuracy(uncoveredy, declist_prediction, label)
            "Binary accuracy for target class $label: $binary_acc"
        end
  

        # all the samples have been covered
        if length(covered_indices) == ninstances(uncoveredX)
            Base.@debug "All the samples have been covered, exiting the training loop"
            break end

        # Remove the covered samples from the uncovered slice of the dataset
        uncovered_slice = setdiff(1:ninstances(uncoveredX), covered_indices)
        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]
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
    !isnothing(max_infogain_ratio) && @assert (0 <= max_infogain_ratio <= 1) "Parameter `max_infogain_ratio` must be in range [0,1], but $(maxpurity_gamma) encountered."

    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter `max_rule_length` cannot be less" *
                                                               "than one. Please provide a valid value."

    @assert (0 < split_ratio < 1) "Parameter `split_ratio` must be in range (0,1)"
    @assert (min_rule_coverage > 0) "Parameter `min_rule_coverage` must be ≥ 1"

    # in Parameters.jl
    searchmethod = reconstruct(searchmethod, kwargs)

    info_dl = (;
        supporting_labels=y,
    )

    y, labels = y |> maptointeger
    poslabel_idx = findfirst(x -> x == poslabel, labels) # indice in labels della classe positiva)

    @assert !isnothing(poslabel_idx) "The dataset provided must contain at least one positive sample!"

    num_instances = ninstances(X)
    @assert length(y) == ninstances(X) "The sizes of the training data X and of the labels y do not match. X has $num_instances instances, whilst y has length $(length(y))"

    uncovered_original_y = y
    y = UInt32.(y .== poslabel_idx)  # ora y è un array di {0,1}^n, dove 1 corrisponde alla classe positiva e 0 ad un'altra

    # samples yet to be covered by any Rule in the RuleSet
    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase_sat_mask = falses(ninstances(X))   # sat mask della rulebase su uncoveredX
    data_curr_ruleset_desc_length = Inf
    dataset_num_selectors = get_num_independent_selectors(X, y, discretizedomain)
    
    # DEBUG
    curr_TDL = get_initial_dataset_bits(y)
    curr_min_TDL = curr_TDL
    
    rulebase = Rule[]
    while true

        if !isnothing(max_rulebase_length) && length(rulebase) >= max_rulebase_length
            @debug "The IREP* loop was stopped because the specified maximum amount of rules ($max_rulebase_length) has been reached"
            break
        end

        split = split_instances(uncoveredX, uncoveredy, uncoveredw, split_ratio, rng)
        split === nothing && break

        num_uncovered_pos = count(label -> label == 1, uncoveredy)    # total number of uncovered positive samples

        if num_uncovered_pos < min_rule_coverage
            Base.@debug "Training converged because the number of positive samples remaining is lower than min_rule_coverage = $min_rule_coverage"
            break end

        bestantecedent = findbestantecedent(searchmethod,
            split.gr...,
            #
            loss_function,
            max_infogain_ratio,
            default_alphabet,
            discretizedomain,
            significance_alpha,
            min_rule_coverage; max_rule_length=max_rule_length,
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
        bestantecedent, bestantecedent_prune_cov = pruneantecedent(bestantecedent, split.pr...)

        # Create the new Rule as an instance of "Rule" from SoleModels
        coverage_indices = compute_global_coverage(bestantecedent, split, bestantecedent_prune_cov)
        rule = build_rule(bestantecedent, uncovered_original_y, poslabel, coverage_indices, labels)


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
        uncovered_slice = setdiff(1:ninstances(uncoveredX), coverage_indices)
        
        # Stop if the entire dataset has been covered, otherwise slicedataset would throw an error
        if length(uncovered_slice) == 0
            break end
        
        
        Base.@debug "Number of uncovered samples remaining: $(length(uncovered_slice))"


        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]
        uncovered_original_y = @view uncovered_original_y[uncovered_slice]
    end

    default_prediction = "other"    # default prediction if no other Rule applies

    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncovered_original_y)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(default_prediction, length(uncovered_original_y)),
    )
    default_consequent = ConstantModel(default_prediction, info_cm)
    return DecisionList(rulebase, default_consequent, info_dl)
end


"""
    Splits the dataset (X,y) with weights w into one part (growX, growy, grow_w) and another (pruneX, pruney) and
    saves the indices of the two datasets in growindxs and pruneindxs.

    Returns a Named Tuple with 
        gr = NamedTuple ( X = growX, y = growy, w = groww ),
        pr = NamedTuple ( X = prunX, y = pruny ),
        gr_inds = growindxs
        permutation = perm_indices 
    Where permutation is a random permutation of numbers from 1 to n (size of X passed as an argument), such that the first
    first ngrow = round(n * split_ratio) indices of permutation are used for the growth set, whilst the others for the
    pruning set.
    growindxs and prunindxs are the indices of the samples in X selected for the growth set and the
    pruning set, respectively.
"""
function split_instances(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::AbstractVector{<:Real},
    split_ratio::Real,
    rng::AbstractRNG = Random.default_rng()
)
    n = ninstances(X)
    ngrow = round(Integer, n * split_ratio)

    # return nothing if the split would put all the data either in the grow category or in the prune category
    if ngrow == 0 || n - ngrow == 0
        return nothing
    end

    permindxs = randperm(rng, n)
    growindxs = permindxs[1:ngrow]
    prunindxs = permindxs[ngrow+1:end]

    growX = slicedataset(X, growindxs, return_view = true)
    growy = y[growindxs]
    groww = w[growindxs]

    prunX = slicedataset(X, prunindxs, return_view = true)
    pruny = y[prunindxs]
    prunw = w[prunindxs]

    return (
        gr=(X=growX, y=growy, w=groww),
        pr=(X=prunX, y=pruny, w=prunw),
        gr_inds=growindxs,
        pr_inds=prunindxs,
        permutation=permindxs
    )
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
    split::NamedTuple,
    bestantecedent_prune_cov::Union{Nothing, Vector{<:Bool}, BitVector}
)
    growX = split.gr.X
    pruneX = split.pr.X

    grow_mask = check(antecedent, growX)
    grow_cov_local = findall(grow_mask)
    grow_cov_global = split.gr_inds[grow_cov_local]


    if bestantecedent_prune_cov === nothing
        prune_mask = check(antecedent, pruneX)
    else
        prune_mask = bestantecedent_prune_cov
    end
    prune_cov_local = findall(prune_mask)
    prune_cov_global = split.pr_inds[prune_cov_local]

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
    uncovered_original_y::AbstractVector{<:UInt32}, 
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

Genera tutte le versioni "potate" (prefissi) della regola `rule`,
in ordine decrescente di lunghezza (dalla regola completa al suo atomo più semplice).

Utile per la fase di pruning di RIPPER.
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
    _best_covmask = check(_best_formula, X)
    _best_score = -1.0   # valore minimo possibile per (p - n)/(p + n)

    # 3. Evaluate all possible pruned versions of the formula 
    for pformula in generate_pruned_formulas(antecedent)

        p_covmask = check(pformula, X)

        p = sum(w[posmask .& p_covmask]) # Sum of True positives weight values
        n = sum(w[negmask .& p_covmask]) # Sum of False positives weight values
        # avoid division by zero or empty rules
        if p + n == 0
            continue
        end

        # v* (IREP* pruning criterion)
        score = (p - n) / (p + n)

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
given the previous satisfaction/coverage mask 'prev_ruleset_satmask' of the ruleset, and a new rule added to the ruleset """
function rs_dataset_bits(
    X::AbstractLogiset, 
    y::AbstractVector{<:UInt32},
    rule::Rule,
    prev_ruleset_satmask::BitVector
)
    n_samples = ninstances(X)

    rule_sat_mask = check(rule.antecedent, X)       # check which samples are covered by the new rule
    ruleset_sat_mask = prev_ruleset_satmask .| rule_sat_mask    # update the sat mask of the whole ruleset by adding samples covered by the new rule

    num_pos = count(label -> label == 1, y)
    p = sum(ruleset_sat_mask)                   # number of samples covered by the ruleset

    ruleset_covered_labels = y[ruleset_sat_mask]
    tp = count(==(1), ruleset_covered_labels)
    fp = count(!=(1), ruleset_covered_labels)

    fn = num_pos - tp       # false negatives

    desc_length = log2binomial(p, fp) + log2binomial(n_samples - p, fn)
    return desc_length, ruleset_sat_mask
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