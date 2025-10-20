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
* `loss_function::Function = soleentropy` is the function that assigns a score to each partial solution.
* `max_infogain_ratio::Real=1.0`: constrains the maximum information gain for anantecedent with respect to the uncovered training set. Its value is bounded between 0 and 1.
* `default_alphabet::Union{Nothing,AbstractAlphabet}=nothing` offers the flexibility to define a tailored alphabet upon which antecedents generation occurs.
* `discretizedomain::Bool=false`:  discretizes continuous variables by identifying optimal cut points
* `significance_alpha::Union{Real,Nothing}=0.0` is the significant alpha
* `min_rule_coverage::Union{Nothing,Integer} = 1` specifies the minimum number of instances covered by each rule.
* `max_rule_length::Union{Nothing,Integer} = nothing` specifies the maximum length allowed for a rule in the search algorithm.
* `loss_function::Function = soleentropy` is the function that assigns a score to each partial solution.
* `max_infogain_ratio::Real=1.0`: constrains the maximum information gain for anantecedent with respect to the uncovered training set. Its value is bounded between 0 and 1.
* `default_alphabet::Union{Nothing,AbstractAlphabet}=nothing` offers the flexibility to define a tailored alphabet upon which antecedents generation occurs.
* `discretizedomain::Bool=false`:  discretizes continuous variables by identifying optimal cut points
* `significance_alpha::Union{Real,Nothing}=0.0` is the significant alpha
* `max_rulebase_length::Union{Nothing,Integer}` is the maximum length of the rulebase;
* `suppress_parity_warning::Bool` if `true`, suppresses parity warnings.
* Any additional keyword argument will be imputed to the `searchmethod`, replacing its original value.

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

    loss_function::Function=ModalDecisionLists.LossFunctions.entropy,
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
        # supporting_weights=w, # TODO
        # supporting_predictions=[],
    )

    instanceset = InstanceSet(X,y,w)
    @show instanceset
    


    y, labels = y |> maptointeger
    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    # TODO
    # instanceset = InstanceSet(X, y, w=nothing)

    rulebase = Rule[]       # Il rulebase effettivo
    while true

        # bestantecedent_coverage è un array di 0 e 1 con 1 negli indici i dove la regola trovata copre il sample xi (in unconveredX)
        bestantecedent, bestantecedent_coverage = findbestantecedent(searchmethod,

            uncoveredX, uncoveredy, uncoveredw,
            #
            loss_function,
            max_infogain_ratio,
            default_alphabet,
            discretizedomain,
            significance_alpha,
            min_rule_coverage;

            max_rule_length = max_rule_length,
            nlabels = length(labels)
        )
        @show length((bestantecedent_coverage))

        bestantecedent == ⊤ && break    # if bestantecedent == ⊤ break end)

        rule = begin
            justcoveredy = uncoveredy[bestantecedent_coverage]
            justcoveredw = uncoveredw[bestantecedent_coverage]
            # indice della classe associata alla regola
            predlabel = SoleModels.bestguess(labels[justcoveredy], justcoveredw; suppress_parity_warning=suppress_parity_warning)
            # prediction = labels[consequent_i]

            # le informazioni che vanno al ConstantModel che sarà il consequent dell'oggetto Rule che rappresenta la regola appena trovata
            info_cm = (;
                supporting_labels=[labels[x] for x in collect(justcoveredy)],       # array con i labels dei sample appena coperti dalla regola creata
                # supporting_weights=collect(justcoveredw), # TODO
                supporting_predictions=fill(predlabel, length(justcoveredy)),       # array dove per ogni sample appena coperto c'è il label che la nuova regola gli assegna 
            )
            consequent = ConstantModel(predlabel, info_cm)

            # info della struct regola appena trovata
            info_r = (;
                supporting_labels=[labels[x] for x in collect(uncoveredy)],
                # supporting_weights=collect(uncoveredw), # TODO
                # supporting_predictions=fill(prediction, length(uncoveredy)),
            )
            Rule(bestantecedent, consequent, info_r)
        end

        push!(rulebase, rule)

        # indici dei samples non ancora coperti dalla nuova regola (e quindi da nessun'altra)
        uncovered_slice = (!).(bestantecedent_coverage)

        # da SoleData
        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]

        if !isnothing(max_rulebase_length) && length(rulebase) > (max_rulebase_length - 1)
            break
        end
    end
    # !allequal(uncoveredy) && @warn "Remaining classes are not all equal; defaultclass represents the best estimate."
    prediction = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    prediction = labels[prediction]
    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncoveredy)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(prediction, length(uncoveredy)),
    )
    defaultconsequent = ConstantModel(prediction, info_cm)
    return DecisionList(rulebase, defaultconsequent, info_dl)
end

function build_cn2(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=BeamSearch(), kwargs...)
end

function build_orange_cn2(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    error("TODO: what's the default parametrization for orange CN2?")
    # return sequentialcovering(X, y, w; searchmethod=BeamSearch(), kwargs...)
end

function build_randcn2(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    w::Union{Nothing,AbstractVector{<:Real},Symbol}=default_weights(length(y));
    kwargs...
)
    return sequentialcovering(X, y, w; searchmethod=RandSearch(), kwargs...)
end





############################################################################################
################### SequentialCovering - RIPPER ######################################
############################################################################################



# TODO: fare una versione solo per il caso binario in modo che sia fedele all'algoritmo originale. 
# Basta aggiungere un parametro pos_label per il label da considerarsi positivo, e poi rimpiazzare y con un
# array di 1 dove y = pos_label e 0 dove y != pos label, tipo "y = y .== pos_label_idx"
function IREP_Star(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    poslabel::CLabel,
    tdl_threshold::Int = 64,
    w::Union{Nothing,AbstractVector{U},Symbol}=default_weights(length(y));
    searchmethod::SearchMethod=BeamSearch(),
    split_ratio::Real=0.66,

    loss_function::Function=ModalDecisionLists.LossFunctions.entropy,
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
    # ARGUMENT PARSING
    !isnothing(max_rulebase_length) && @assert max_rulebase_length > 0 "`max_rulebase_length` must be  > 0"
    max_rulebase_length = (isnothing(max_rulebase_length)) ? Inf : max_rulebase_length


    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]
    @assert (0 <= max_infogain_ratio <= 1) "max_infogain_ratio must be in range [0,1], but $(maxpurity_gamma) encountered."

    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                "than one. Please provide a valid value."

    w = if isnothing(w) || w == :default
        default_weights(y) # ones
    elseif w == :rebalance
        balanced_weights(y)
    else
        w
    end

    # in Parameters.jl
    searchmethod = reconstruct(searchmethod, kwargs)

    !(ninstances(X) == length(y)) && error("Mismatching number of instances between X and y! ($(ninstances(X)) != $(length(y)))")
    !(ninstances(X) == length(w)) && error("Mismatching number of instances between X and w! ($(ninstances(X)) != $(length(w)))")
    (ninstances(X) == 0) && error("Empty trainig set")

    info_dl = (;
        supporting_labels=y,
        # supporting_weights=w, # TODO
        # supporting_predictions=[],
    )


    # y è un vettore di interi {1,2,...} corrispondenti ai label, labels è un vettore di clabel come {"setosa", "virginica", "versicolor"}
    y, labels = y |> maptointeger   
    poslabel_idx = findfirst(x -> x == poslabel, labels) # indice in labels della classe positiva

    uncovered_original_y = y

    y = convert.(UInt32, (y .== poslabel_idx))  # ora y è un array di {0,1}^n, dove 1 corrisponde alla classe positiva e 0 ad un'altra

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]       # Il rulebase effettivo
    rulebase_sat_mask = falses( ninstances(X) )   # sat mask della rulebase su uncoveredX
    data_curr_ruleset_desc_length = Inf
    println("Entering main IREP* loop...")
    
    i = 1

    while length(rulebase) < max_rulebase_length
        println("------------------- Starting iteration #$(i) -------------------")
        
        result = split_instances(uncoveredX, uncoveredy, uncoveredw, split_ratio)
        result === nothing && break
        growX, growy, grow_w, pruneX, pruney, grow_inds, prune_inds = result

        # prima era una LeftmostConjunctiveForm
        bestantecedent = findbestantecedent(searchmethod,
            growX, growy, grow_w,
            #
            loss_function,
            max_infogain_ratio,
            default_alphabet,
            discretizedomain,
            significance_alpha,
            min_rule_coverage;

            max_rule_length = max_rule_length,
            nlabels = 2
        )
        coverage = bestantecedent.covmask
        bestantecedent = bestantecedent.formula

        println("best antecedent: $bestantecedent")
        bestantecedent == ⊤ && break

        bestantecedent, bestantecedent_prune_cov = PruneRule(pruneX, pruney, bestantecedent)

        coverage_indices = compute_coverage(bestantecedent, growX, grow_inds, pruneX, prune_inds, bestantecedent_prune_cov)

        # costruisce l'istanza di Rule da utilizzare nella DecisionList che si ritorna con Sole
        rule = build_rule(bestantecedent, uncovered_original_y, poslabel, coverage_indices, labels)
        
        # Check TDL
        rule_desc_length = _r_theory_bits(X, y, rule, discretizedomain)

        push!(rulebase, rule)

        data_new_ruleset_desc_length, rulebase_sat_mask = rs_dataset_bits(X, y, rule, rulebase_sat_mask)

        println("New Rule description length: $rule_desc_length")
        println("New dataset description length: $data_new_ruleset_desc_length")

        # ΔTDL = ΔTDL(Ruleset) + ΔTDL(Dataset | Ruleset), dove ΔTDL(Ruleset) = TDL(Ruleset + Rule_i) - TDL(Ruleset) = TDL(Rule_i), 
        # # a ogni iterazione si aggiunge una regola e quindi anche la tdl del ruleset aumenta della lunghezza di descrizione della regola
        ΔTDL_data_given_ruleset = data_new_ruleset_desc_length - data_curr_ruleset_desc_length
        ΔTDL_ruleset = rule_desc_length                         
        ΔTDL = ΔTDL_ruleset + ΔTDL_data_given_ruleset
        println("Total tdl difference: $ΔTDL")

        if ΔTDL > tdl_threshold
            pop!(rulebase)
            break
        end

        data_curr_ruleset_desc_length = data_new_ruleset_desc_length


        # Rimozione
        # Incapsulare
        uncovered_slice = setdiff(1:ninstances(uncoveredX), coverage_indices)
        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]
        uncovered_original_y = @view uncovered_original_y[uncovered_slice]

        println("uncovered_slice length: $(length(uncovered_slice))")

        i += 1
        println("------------------- End of iteration #$(i) -------------------")
    end


    #prediction = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    #prediction = (prediction == 1) ? poslabel : "other";
    
    prediction = "other"    # default prediction se nessuna altra regola si applica


    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncovered_original_y)],
        # supporting_weights=collect(justcoveredw), # TODO
        supporting_predictions=fill(prediction, length(uncovered_original_y)),
    )
    defaultconsequent = ConstantModel(prediction, info_cm)
    return DecisionList(rulebase, defaultconsequent, info_dl)
end


"""
    Separa il dataset (X,y) con i pesi w in una parte (growX, growy, grow_w) e in un'altra (pruneX, pruney) dove
    gli indici dei due dataset sono salvati in grow_inds e in prune_inds
"""
function split_instances(X, y, w, split_ratio)
    n = ninstances(X)
    n_grow = convert(Int, ceil(n * split_ratio))
    # @Edo: Quando questo test è verificato
    if n_grow == 0 || n - n_grow == 0
        return nothing
    end
    all_inds = collect(1:n)
    grow_inds = randperm(n)[1:n_grow]
    prune_inds = setdiff(all_inds, grow_inds)
    
    growX = slicedataset(X, grow_inds)
    growy = y[grow_inds]
    groww = w[grow_inds]

    pruneX = slicedataset(X, prune_inds)
    pruney = y[prune_inds]

    return growX, growy, groww, pruneX, pruney, grow_inds, prune_inds
end

"""
    Calcola gli indici dei sample del dataset originale che sono coperti dalla condizione antecedent.
    Se growX e pruneX sono i sottoinsiemi del dataset X tali che growX = X[grow_inds] e pruneX = X[prune_inds],
    la funzione ritorna l'insieme di indici tali che X[inds] sono i samples coperti da antecedent
"""
function compute_coverage(
    antecedent::LeftmostConjunctiveForm, 
    growX, grow_inds, 
    pruneX, prune_inds, 
    bestantecedent_prune_cov
)    
    grow_mask = check(antecedent, growX)
    grow_cov_local = findall(grow_mask)
    grow_cov_global = grow_inds[grow_cov_local]

    
    if bestantecedent_prune_cov === nothing 
        prune_mask = check(antecedent, pruneX)
    else 
        prune_mask = bestantecedent_prune_cov
    end
    prune_cov_local = findall(prune_mask)
    prune_cov_global = prune_inds[prune_cov_local]
    
    return vcat(grow_cov_global, prune_cov_global)
end


"""
    Crea un'istanza di Rule con il dato antecedent, calcolando il label più frequente tra i sample coperti dall'antecedent.
    antecedent --> antecedente della regola in questione
    labels --> i label del dataset utilizzato per costruire la regola (istanze di CLabel)
    uncoveredy --> vettore numerico di interi corrispondenti alle classi dei samples
    uncoveredw --> vettore di reali con i pesi dei vari samples
    coverage_indices --> indici dei valori che la regola ha coperto
"""
function build_rule(antecedent, uncovered_original_y, poslabel, coverage_indices, labels)
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



function PruneRule(
    pruneX::AbstractLogiset,
    pruneY::Vector{UInt32},
    rule::LeftmostConjunctiveForm
)
    n_conditions = length(rule.grandchildren)  # cercare funzione default di sole (tipo natoms?)

    pos_samples_indxs = findall(label -> label == 1, pruneY)
    neg_samples_indxs = findall(label -> label != 1, pruneY)

    best_rule = rule
    best_rule_score = -Inf
    best_rule_sat_mask = []
    
    # per ogni sottoinsieme finale non nullo delle condizioni
    for cond_idx = n_conditions:-1:1
        new_grandchildren = rule.grandchildren[1:cond_idx]
        new_rule = LeftmostConjunctiveForm(new_grandchildren)

        rule_sat_mask = check(new_rule, pruneX)  
        rule_covered_idxs = findall(rule_sat_mask)

        p = length(intersect(rule_covered_idxs, pos_samples_indxs)) # num. di samples positivi coperti dalla regola
        n = length(intersect(rule_covered_idxs, neg_samples_indxs)) # num. di samples negativi coperti dalla regola
        
        rule_score = (p - n)/(p + n)        # v* nel paper
        if rule_score > best_rule_score
            best_rule_score = rule_score
            best_rule = new_rule
            best_rule_sat_mask = rule_sat_mask
        end
    end

    return best_rule, best_rule_sat_mask
end


"""
    Ritorna la lista di tutti gli antecedenti possibili con una sola condizione dall'alfabeto a, escludendo
    condizioni equivalenti (ovvero quelle che coprono le stesse istanze di X)
"""
function unaryconditions_noneq(
    a::UnionAlphabet,
    X::AbstractLogiset
)::Vector{Tuple{Atom{ScalarCondition},SatMask}}
    seen_masks = Set{BitVector}()
    conditions = Tuple{Atom{ScalarCondition},SatMask}[]
    for univalph in subalphabets(a)
        for atom in atoms(univalph)
            mask = check(atom, X)
            if mask ∉ seen_masks
                push!(conditions, (atom, mask))
                push!(seen_masks, mask)
            end
        end
    end
    return conditions
end


"""
    Ritorna la TDL (Total Description Length) di una regola in forma di LeftmostConjunctiveForm per un certo dataset (X,y)
"""
function _r_theory_bits(X::AbstractLogiset, y, rule::Rule, discretizedomain::Bool = false)
    alph = alphabet(X;
        discretizedomain = discretizedomain,
        y = y
    )

    #conds = unaryconditions(conjuncts_search_method, alph, X)
    conds = unaryconditions_noneq(alph, X)
    #println("------ N CALCOLATO : n = $(length(conds))")

    n = length(conds)
    k = 1 + nconnectives(rule.antecedent) # si assume che la formula di Rule sia una LeftmostConjunctiveForm
    pr = k / n

    S = k * log2(1/pr) + (n - k) * log2(1/(1 - pr))
    K = log2(k)
    desc_length = (S + K) * 0.5

    #println("n = $n | k = $k | natoms: $(natoms(rule.antecedent)) | nleaves: $(nleaves(rule.antecedent))")
    return max(desc_length, 1)
end


"""
    Ritorna la TDL (Total Description Length) di un ruleset per un certo dataset (X,y)
"""
function _rs_theory_bits(X::AbstractLogiset, y, ruleset::Vector{Rule}, discretizedomain::Bool = false)
    alph = alphabet(X;
        discretizedomain = discretizedomain,
        y = y
    )

    #conds = unaryconditions(conjuncts_search_method, alph, X)
    conds = unaryconditions_noneq(alph, X)
    #println("------ N CALCOLATO : n = $(length(conds))")

    total_desc_length = 0
    n = length(conds)
    for rule ∈ ruleset
        k = 1 + nconnectives(rule.antecedent) # si assume che la formula di Rule sia una LeftmostConjunctiveForm
        pr = k / n

        S = k * log2(1/pr) + (n - k) * log2(1/(1 - pr))
        K = log2(k)
        desc_length = (S + K) * 0.5
        total_desc_length += desc_length
    end

    #println("n = $n | k = $k | natoms: $(natoms(rule.antecedent)) | nleaves: $(nleaves(rule.antecedent))")
    return max(total_desc_length, 1)
end


# ritorna un'approssimazione di ln(n!) usando Stirling
function log2_factorial(n::Int)::Real
    if n == 0
        return 0
    end
    
    println("[log2_factorial] n = $n")
    return max(0, 0.5 * (1 + log2(π * n)) + n * log2(n/ℯ))
end

# ritorna un'approssimazione di ln( n choose k ) usando log2_factorial
function log2binomial(n::Int, k::Int)::Real
    if k == 0
        return 0
    end

    # Per n piccoli, approssimazione Stirling è imprecisa (es. per n=5, log2(120)≈6.9, Stirling≈7.1). Per precisione, usa log2(factorial(big(n))) per n<20, Stirling per grandi.

    println("[log2_binomial] n = $n | k = $k")
    return log2_factorial(n) - log2_factorial(k) - log2_factorial(n-k)
end


# @Edo TODO: Qui lavoriamo su Bitmask, più efficente
function rs_dataset_bits(
    X::AbstractLogiset, y, 
    rule::Rule, 
    prev_ruleset_satmask::AbstractVector{Bool}
)
    n_samples = ninstances(X)

    rule_sat_mask = check(rule.antecedent, X)       # controlla quali sample copre la nuova regola
    ruleset_sat_mask = prev_ruleset_satmask .| rule_sat_mask    # aggiorno la maschera dei sample coperti dalle regole
    
    ruleset_covered_idxs = findall(ruleset_sat_mask)

    pos_samples_indxs = findall(label -> label == 1, y)
    neg_samples_indxs = findall(label -> label != 1, y)
    num_pos = length(pos_samples_indxs)

    p = length(ruleset_covered_idxs)
    tp = length(intersect(ruleset_covered_idxs, pos_samples_indxs)) # num. di samples positivi coperti dalla regola 
    fp = length(intersect(ruleset_covered_idxs, neg_samples_indxs)) # false positives
    
    fn = num_pos - tp  # false negatives
    
    println("[rs dataset bits] n_samples: $n_samples | num_pos: $num_pos | valori coperti: $p | false positives: $fp | false negatives: $fn")

    #desc_length = log2( binomial(p, fp) ) + log2( binomial( n_samples - p, fn ) )   # va in overflow
    desc_length = log2binomial(p, fp) + log2binomial(n_samples - p, fn)
    return desc_length, ruleset_sat_mask
end
