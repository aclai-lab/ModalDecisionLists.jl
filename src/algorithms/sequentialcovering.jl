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
    )

    y, labels = y |> maptointeger
    uncoveredX = X
    uncoveredy = y
    uncoveredw = w

    rulebase = Rule[]       # Il rulebase effettivo
    while true

        # bestantecedent_coverage è un array di 0 e 1 con 1 negli indici i dove la regola trovata copre il sample xi (in unconveredX)
        bestantecedent = findbestantecedent(searchmethod,

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
    prediction = SoleModels.bestguess(uncoveredy; suppress_parity_warning = suppress_parity_warning)
    prediction = labels[prediction]
    info_cm = (;
        supporting_labels=[labels[x] for x in collect(uncoveredy)],
        supporting_predictions=fill(prediction, length(uncoveredy)),
    )
    defaultconsequent = ConstantModel(prediction, info_cm)
    return DecisionList(rulebase, defaultconsequent, info_dl)
end


############################################################################################
################### SequentialCovering - RIPPER ######################################
############################################################################################



# TODO: fare una versione solo per il caso binario in modo che sia fedele all'algoritmo originale. 
# Basta aggiungere un parametro pos_label per il label da considerarsi positivo, e poi rimpiazzare y con un
# array di 1 dove y = pos_label e 0 dove y != pos label, tipo "y = y .== pos_label_idx"
function irepstar(
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

    !isnothing(max_rulebase_length) && @assert max_rulebase_length > 0 "`max_rulebase_length` must be  > 0"

    @assert w isa AbstractVector || w in [nothing, :rebalance, :default]
    @assert (0 <= max_infogain_ratio <= 1) "max_infogain_ratio must be in range [0,1], but $(maxpurity_gamma) encountered."

    !isnothing(max_rule_length) && @assert max_rule_length > 0 "Parameter 'max_rule_length' cannot be less" *
                                                "than one. Please provide a valid value."
    # in Parameters.jl
    searchmethod = reconstruct(searchmethod, kwargs)

    info_dl = (;
        supporting_labels=y,
    )

    y, labels = y |> maptointeger   
   poslabel_idx = findfirst(x -> x == poslabel, labels) # indice in labels della classe positiva

    uncovered_original_y = y

    y = UInt32.(y .== poslabel_idx)  # ora y è un array di {0,1}^n, dove 1 corrisponde alla classe positiva e 0 ad un'altra

    uncoveredX = X
    uncoveredy = y
    uncoveredw = w


    rulebase_sat_mask = falses( ninstances(X) )   # sat mask della rulebase su uncoveredX
    data_curr_ruleset_desc_length = Inf

    dataset_num_selectors = get_num_independent_selectors(X, y, discretizedomain)

    rulebase = Rule[]
    while true

        if !isnothing(max_rulebase_length) && length(rulebase) < max_rulebase_length
            break
        end

        split = split_instances(uncoveredX, uncoveredy, uncoveredw, split_ratio)
        split === nothing && break

        bestantecedent = findbestantecedent(searchmethod,
            split.gr..., 
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

        istop(bestantecedent) && break
        target_class = 1

        bestantecedent = pruneantecedent(bestantecedent, split.pr...)

        # @Nicola TODO: Continua qui....

        # costruisce l'istanza di Rule da utilizzare nella DecisionList che si ritorna con Sole
        rule = build_rule(bestantecedent, uncovered_original_y, poslabel, coverage_indices, labels)
        
        # Check TDL
        rule_desc_length = _r_theory_bits(rule, dataset_num_selectors)

        push!(rulebase, rule)

        data_new_ruleset_desc_length, rulebase_sat_mask = rs_dataset_bits(X, y, rule, rulebase_sat_mask)

        println("New Rule description length: $rule_desc_length")
        println("New dataset description length: $data_new_ruleset_desc_length")

        # ΔTDL = ΔTDL(Ruleset) + ΔTDL(Dataset | Ruleset), dove ΔTDL(Ruleset) = TDL(Ruleset + Rule_i) - TDL(Ruleset) = TDL(Rule_i), 
        # # a ogni iterazione si aggiunge una regola e quindi anche la tdl del ruleset aumenta della lunghezza di descrizione della regola
        ΔTDL_data_given_ruleset = data_new_ruleset_desc_length - data_curr_ruleset_desc_length
        ΔTDL_ruleset = rule_desc_length                         
        ΔTDL = ΔTDL_ruleset + ΔTDL_data_given_ruleset

        # println("Total tdl difference: $ΔTDL")

        if ΔTDL > tdl_threshold
            pop!(rulebase)
            break
        end

        data_curr_ruleset_desc_length = data_new_ruleset_desc_length

        # Rimozione
        # Incapsulare
        uncovered_slice = setdiff(1:ninstances(uncoveredX), coverage_indices)
        # tutto il dataset è stato coperto, evitiamo di tirare un errore su slicedataset
        if length(uncovered_slice) == 0 
            break
        end
        println("uncovered_slice length: $(length(uncovered_slice))")

        uncoveredX = slicedataset(uncoveredX, uncovered_slice; return_view=true)
        uncoveredy = @view uncoveredy[uncovered_slice]
        uncoveredw = @view uncoveredw[uncovered_slice]
        uncovered_original_y = @view uncovered_original_y[uncovered_slice]

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
    gli indici dei due dataset sono salvati in growindxs e in pruneindxs
"""
function split_instances(X, y, w, split_ratio)
# function split_instances(X, y, w, split_ratio, seed)

    n = ninstances(X)
    ngrow = convert(Int, ceil(n * split_ratio))
    # @Nicola Semplicemente round(n * split_ratio)

    # @Nicola TODO: Cambiare, bisogna fare un test a priori 
    # su split_ratio (che non sia saturo, quindi non 0 o 1)
    if ngrow == 0 || n - ngrow == 0
        return nothing
    end

    # TODO @Nicola : Aggungere seed per riproducibilità !!

    # ho cambiato così
    permindxs = randperm(n)
    growindxs = permindxs[1:ngrow]
    prunindxs = permindxs[ngrow+1:end]

    growX = slicedataset(X, growindxs)
    growy = y[growindxs]
    groww = w[growindxs]

    prunX = slicedataset(X, prunindxs)
    pruny = y[prunindxs]

    return (
        gr = (X = growX, y = growy, w = groww),
        pr = (X = prunX, y = pruny),
        permutation = permindxs
    )
end

"""
    compute_coverage(
        antecedent::LeftmostConjunctiveForm, 
        growX, growindxs, 
        pruneX, pruneindxs, 
        bestantecedent_prune_cov
    ) -> Vector{Int}

Calcola gli indici **globali** delle istanze del dataset originale che sono 
coperte da un dato `antecedent` (regola logica o congiunzione di condizioni), 
combinando la copertura nei sottoinsiemi *grow* e *prune*.

# Descrizione
il dataset viene suddiviso in due parti:
- **grow set** (`growX`): usato per costruire la regola;
- **prune set** (`pruneX`): usato per ottimizzarla e ridurne l'overfitting.

Questa funzione valuta la regola su entrambi i sottoinsiemi e restituisce 
gli indici delle istanze (riferiti al dataset originale) che risultano coperte.

Se è già disponibile una maschera di copertura per la fase di pruning 
(`bestantecedent_prune_cov`), questa viene riutilizzata per evitare 
ricomputazioni.

# Argomenti
- `antecedent::LeftmostConjunctiveForm`: la regola o congiunzione di condizioni da valutare.
- `growX`: il sottoinsieme di dati usato nella fase di *growing*.
- `growindxs`: vettore degli indici globali corrispondenti alle istanze di `growX`.
- `pruneX`: il sottoinsieme di dati usato nella fase di *pruning*.
- `pruneindxs`: vettore degli indici globali corrispondenti alle istanze di `pruneX`.
- `bestantecedent_prune_cov`: maschera booleana opzionale già calcolata che indica 
  le istanze coperte su `pruneX` (può essere `nothing`).

# Ritorna
- `Vector{Int}` — un vettore contenente tutti gli indici **globali** delle istanze 
  coperte dall'antecedent, sia in `growX` che in `pruneX`.



TODO: Tutti i commenti in inglese
"""


# @Nicola: questa funzione non mi piace tanto, intuisco ci 
# sia un modo migliore di farla che evita una ulteriore check.
# Io partirei dalla funzione split_instances(...). Qui so come vengono 
# permutate le istanze, magari posso portarmi dietro questa info e riuscire a riordinare tutto
function compute_global_coverage(
    antecedent::Antecedent, 
    growX, growindxs, 
    pruneX, pruneindxs, 
    bestantecedent_prune_cov
)    
    grow_mask = check(antecedent.formula, growX)
    grow_cov_local = findall(grow_mask)
    grow_cov_global = growindxs[grow_cov_local]

    
    if bestantecedent_prune_cov === nothing 
        prune_mask = check(antecedent.formula, pruneX)
    else 
        prune_mask = bestantecedent_prune_cov
    end
    prune_cov_local = findall(prune_mask)
    prune_cov_global = pruneindxs[prune_cov_local]
    
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





"""
    generate_pruned_rules(rule::LeftmostConjunctiveForm)

Genera tutte le versioni "potate" (prefissi) della regola `rule`,
in ordine decrescente di lunghezza (dalla regola completa al suo atomo più semplice).

Utile per la fase di pruning di RIPPER.
"""

# TODO: @Nicola: specificare un ulteriore parametro per il pruning: 
# Esistono metodi alternativi oper il pruning invece che rimuovere in maniera monotona l'ultima condizione ? 
function generate_pruned_formulas(ant::Antecedent)
    _range = nconds(ant):-1:1
    return [LeftmostConjunctiveForm(conds(ant)[1:i]) 
        for i in _range
    ]
end

function pruneantecedent(antecedent::Antecedent,
    X::AbstractLogiset,
    y::Vector{UInt32},
)
    target_class = 1

    # 1. Costruzione delle maschere positive/negative rispetto alla classe target
    posmask = y .== target_class
    negmask = .!posmask

    # 2. Inizializzazione del miglior antecedente (best rule)
    _best_formula = antecedent.formula
    _best_covmask = check(_best_formula, X)
    _best_score = -1.0   # valore minimo possibile per (p - n)/(p + n)

    # 3. Valuta tutte le versioni potate della formula
    for pformula in generate_pruned_formulas(antecedent)

        p_covmask = check(pformula, X)

        p = sum(posmask .& p_covmask)
        n = sum(negmask .& p_covmask)
        # evita divisioni per zero o regole vuote
        if p + n == 0
            continue
        end

        # v* (RIPPER pruning criterion)
        score = (p - n) / (p + n)

        if score > _best_score
            _best_formula = pformula
            _best_covmask = p_covmask
            _best_score = score
        end
    end

    # 4. Costruzione della maschera di copertura globale prima di istanziare il nuovo antecedente
    # total_cov = falses(length(prunindxs) + length(growindxs))
    # total_cov[growindxs] .= antecedent.covmask          # coverage del growing set
    # total_cov[prunindxs] .= _best_covmask               # coverage del pruning set

    return Antecedent(_best_formula, total_cov)
end


# function pruneantecedent(
#     X::AbstractLogiset,
#     y::Vector{UInt32},
#     antecedent::Antecedent
#     # rule :: Lmcf
# )
#     # Quindi qui la y è booleana ? 
#
#     @show y
#     pos_indxs = findall(label -> label == 1, y)
#     neg_indxs = findall(label -> label != 1, y)
#
#     bestf = antecedent.formula
#     bestf_score = -Inf
#     bestf_satmask = []
#
#     # per ogni sottoinsieme finale non nullo delle condizioni
#     for pformula in generate_pruned_formulas(antecedent)
#
#         satmask = check(pformula, X)  
#
#         rule_covered_idxs = findall(satmask)
#
#         # num. di samples positivi coperti dalla regola
#         p = length(intersect(rule_covered_idxs, pos_indxs)) 
#         # num. di samples negativi coperti dalla regola
#         n = length(intersect(rule_covered_idxs, neg_indxs))
#
#         rule_score = (p - n)/(p + n)        # v* nel paper
#         if rule_score > best_rule_score
#             best_rule_score = rule_score
#             best_rule = new_rule
#             best_rule_sat_mask = rule_sat_mask
#         end
#     end
#
#     return best_rule, best_rule_sat_mask
# end


# Non più necessario, uso direttamente alphabet2conditions che ora filtra le condizioni equivalenti 
#"""
#    Ritorna la lista di tutti gli antecedenti possibili con una sola condizione dall'alfabeto a, escludendo
#    condizioni equivalenti (ovvero quelle che coprono le stesse istanze di X)
#"""
#function unaryconditions_noneq(    
#    a::UnionAlphabet,
#    X::AbstractLogiset
#)::Vector{Tuple{Atom{ScalarCondition},SatMask}}
#    seen_masks = Set{BitVector}()
#    conditions = Tuple{Atom{ScalarCondition},SatMask}[]
#    for univalph in subalphabets(a)
#        for atom in atoms(univalph)
#            mask = check(atom, X)
#            if mask ∉ seen_masks
#                push!(conditions, (atom, mask))
#                push!(seen_masks, mask)
#            end
#        end
#    end
#    return conditions
#end

function get_num_independent_selectors(X::AbstractLogiset, y, discretizedomain::Bool = false)::Int
    alph = alphabet(X;
        discretizedomain = discretizedomain,
        y = y
    )

    independent_conds = alphabet2conditions(AtomGenerator(), alph, X)
    return length(independent_conds)
end

"""
    function _r_theory_bits(rule::Rule, n_possible_conds::Int)::Int

    Ritorna la TDL (Total Description Length) di una regola in forma di LeftmostConjunctiveForm per un certo dataset (X,y)
"""
function _r_theory_bits(rule::Rule, n::Int)
    # conds = unaryconditions_noneq(alph, X)
    # n_old = length(conds) 
    
    #println("\t Conds unaryconds_noneq: $n_old | Conds alphabet2conditions: $n")

    k = 1 + nconnectives(rule.antecedent) # si assume che la formula di Rule sia una LeftmostConjunctiveForm
    pr = k / n

    S = k * log2(1/pr) + (n - k) * log2(1/(1 - pr))
    K = log2(k)
    desc_length = (S + K) * 0.5

    #println("n = $n | k = $k | natoms: $(natoms(rule.antecedent)) | nleaves: $(nleaves(rule.antecedent))")
    return max(desc_length, 1)
end


#"""
#    Ritorna la TDL (Total Description Length) di un ruleset per un certo dataset (X,y)
#    (Non realmente necessario)
#"""
#function _rs_theory_bits(X::AbstractLogiset, y, ruleset::Vector{Rule}, discretizedomain::Bool = false)Comuqnue 
#    alph = alphabet(X;
#        discretizedomain = discretizedomain,
#        y = y
#    )
#
#    #conds = unaryconditions(conjuncts_search_method, alph, X)
#    conds = unaryconditions_noneq(alph, X)
#    #println("------ N CALCOLATO : n = $(length(conds))")
#
#    total_desc_length = 0
#    n = length(conds)
#    for rule ∈ ruleset
#        k = 1 + nconnectives(rule.antecedent) # si assume che la formula di Rule sia una LeftmostConjunctiveForm
#        pr = k / n
#
#        S = k * log2(1/pr) + (n - k) * log2(1/(1 - pr))
#        K = log2(k)
#        desc_length = (S + K) * 0.5
#        total_desc_length += desc_length
#    end
#    
#    #println("n = $n | k = $k | natoms: $(natoms(rule.antecedent)) | nleaves: $(nleaves(rule.antecedent))")
#    return max(total_desc_length, 1)
#end


# ritorna un'approssimazione di ln(n!) usando Stirling
function log2_factorial(n::Int)::Real
    if n == 0
        return 0
    end
    
    println("[log2_factorial] n = $n")
    return max(0, 0.5 * (1 + log2(π * n)) + n * log2(n/ℯ) + 0.115/n)
end

# ritorna un'approssimazione di ln( n choose k ) usando log2_factorial
function log2binomial(n::Int, k::Int)::Real
    if k == 0
        return 0
    end

    # 0.115/n è stato scelto perchè, senza cambiare l'uguaglianza asintotica di Stirling, esegue una correzione
    # piuttosto buona per n piccolo, in questa maniera non serve realmente fare un if n < n_min per usare il fattoriale su piccoli valori
    # Già per n = 1 l'errore assoluto di questa funzione rispetto al valore corretto è 0.00194 e va diminuendo
    # confrontare le due curve in una calcolatrice grafica per farsi un'idea
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
