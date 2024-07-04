# This is a baseline implementation of CN2, drawing inspiration from Orange's implementation
# Author: @edo-007

using SoleData
using SoleLogics
using SoleBase
using SoleModels
using DataFrames
using StatsBase: countmap
import SoleLogics: LogicalInstance, Formula, LeftmostLinearForm
import SoleModels: Rule, AbstractModel, ConstantModel
import SoleBase: CLabel
import SoleData: VariableValue
import SoleData: features

struct Selector
    att::Symbol # simil Feature
    val
    test_operator
end

# >>>

    Base.show(io::IO, s::Selector) = print(io, "($(s.att) $(s.test_operator) $(s.val))")
    test_operator(sel::Selector) = sel.test_operator
    varname(sel::Selector) = sel.att
    treshold(sel::Selector) = sel.val


    function selector2soleatom(sel::Selector)::Atom{ScalarCondition}
        feature = VariableValue(varname(sel))
        tresh = treshold(sel)
        test_op = test_operator(sel)
        return Atom(ScalarCondition(feature, test_op, tresh))
    end
#######################################################################################################

struct RuleBody
    selectors::Vector{Selector}
end

# >>>

    function Base.:(==)(r1::RuleBody, r2::RuleBody)
        if ( length(r1.selectors) != length(r2.selectors) )
                return false
        end
        for r1sel in r1.selectors
            if r1sel ∉ r2.selectors
                return false
            end
        end
            return true
    end

    function Base.show(io::IO, r::RuleBody)
        nselector = length(selectors(r))
        count = 1
        for sel in r.selectors
            print(io, "$sel")
            if count < nselector
                print(io, " ∧ ")
            end
            count = count + 1
        end
    end

    selectors(rb::RuleBody) = rb.selectors
    pushselector!(rule::RuleBody, selector::Selector) = push!(rule.selectors, selector)
    getattributes(r::RuleBody) = [varname(sel) for sel ∈ selectors(r)]

    function base_interpret(
        antecedent::RuleBody,
        X::AbstractDataFrame
    )::Vector{Bool}

        antecedent_coverage = ones(Bool, nrow(X))

        for selector ∈ selectors(antecedent)
            test_op = test_operator(selector)
            selector_coverage = test_op.(X[:, varname(selector)],
                                        treshold(selector)
                                            )
            antecedent_coverage = antecedent_coverage .& selector_coverage
        end
        return antecedent_coverage
    end

    function find_new_selectors(
        X::AbstractDataFrame,
        candidate_antecedent::RuleBody=RuleBody([])
    )::Vector{Selector}
        # candidate_antecedent == [] => sat_indexes = [1,1,1,...,1]
        sat_indexes = base_interpret(candidate_antecedent, X)
        X_covered = X[sat_indexes, :]
        # @show X_covered

        selectorlist = Vector{Selector}([])
        test_ops = [≤, ≥]
        attributes = names(X)
        for attribute ∈ attributes, test_op ∈ test_ops
            map( x -> push!( selectorlist, Selector(Symbol(attribute), x, test_op)),
                        sort(unique(X_covered[:, attribute]))
                )
        end
        possible_selectorlist = [_s for _s in selectorlist if _s ∉ selectors(candidate_antecedent)]
        return possible_selectorlist
    end

############################################################################################

struct MyRule
    body::RuleBody
    head
end

# >>>

    function Base.show(io::IO, rule::MyRule)
        println(io, " $(rule.body) ⟶ ($(rule.head))" )
    end
    head(r::MyRule) = r.head
    body(r::MyRule) = r.body
    selectors(mr::MyRule) = selectors(body(mr))

    function convert_solerule(
        antecedent::RuleBody,
        consequent
    )::Rule
        _atoms = selector2soleatom.(selectors(antecedent))
        sole_antecedent = length(_atoms) > 0 ? LeftmostConjunctiveForm(_atoms) :
                                                    LeftmostConjunctiveForm([⊤])
        sole_consequent = ConstantModel(consequent)
        return Rule(sole_antecedent, sole_consequent)
    end

############################################################################################

struct Star
    antecedents::Array{RuleBody}
end

# >>>

    starsize(star::Star) = length(star.antecedents)
    pushrule!(star, rule) = push!(star.antecedents, rule)
    rulebodies2star(ruleslist::Array{RuleBody}) = Star(ruleslist)
    antecedents(star::Star) = star.antecedents
    selectors(star::Star) = star.antecedents


    function base_refine_antecedents(
        star::Star,
        X::AbstractDataFrame
    )::Star
        if starsize(star) == 0
            possible_selectors = find_new_selectors(X)
            possible_antecedents = map(sel->RuleBody([sel]), possible_selectors)

            return Star(possible_antecedents)
        else
            refined_antecedents = Vector{RuleBody}([])
            for _antecedent in selectors(star)
                possible_selectors = find_new_selectors(X, _antecedent)
                isempty(possible_selectors) &&
                    return Star([])
                for _selector in possible_selectors
                    new_antecedent = deepcopy(_antecedent)
                    pushselector!(new_antecedent, _selector)
                    push!(refined_antecedents, new_antecedent)
                end
            end
        end
        return Star(refined_antecedents)
    end

    Base.isempty(star::Star) = (length(star.antecedents) == 0)

    function Base.show(io::IO, s::Star)
        println("Star:")
        for r in antecedents(s)
            print(" • ")
            println(io,r)
        end
    end
    function Base.show(io::IO, dict::Dict{RuleBody, Float32})
        for key ∈ keys(dict)
            println(io,"$key    $(dict[key])")
        end
    end

function entropy(x)

    length(x) == 0 &&
        return Inf
    v = values(countmap(x))
    (logbase = length(v)) == 1 &&
        return 0.0
    prob = v ./ sum(v)
    return -sum( prob .* log.(logbase, prob) )
end

function base_sortedantecedents(
    star::Star,
    X::AbstractDataFrame,
    y::AbstractVector{<:CLabel},
    beam_width::Int64
)
    starsize(star) == 0 && return [], Inf

    antd_entropies = map( sel->begin
        sat_indexes = base_interpret(sel, X)
        entropy(y[sat_indexes])
    end, antecedents(star))

    i_bests = partialsortperm(antd_entropies, 1:min(beam_width, length(antd_entropies)))
    bestentropy = antd_entropies[i_bests[1]]
    return (i_bests, bestentropy)
end



function base_beamsearch(
    X::AbstractDataFrame,
    y::AbstractVector{<:CLabel},
    beam_width::Integer
)
    best_antecedent = nothing
    bestentropy = entropy(y)

    newstar = Star([])
    while true

        (star, newstar) = newstar, Star([])
        newstar = base_refine_antecedents(star, X)
        # Exit condition
        isempty(newstar) && break

        _sortperm, newbestentropy = base_sortedantecedents(newstar, X, y, beam_width)
        newstar = antecedents(newstar)[_sortperm] |> Star

        if newbestentropy < bestentropy
            best_antecedent = newstar.antecedents[begin]
            bestentropy = newbestentropy
        end
    end
    return best_antecedent
end

function mostcommonvalue(classlist)
    occurrence = countmap(classlist)
    return findmax(occurrence)[2]
end


function build_base_cn2(
    X::AbstractDataFrame,
    y::AbstractVector{<:CLabel};
    beam_width = 3
)
    length(y) != nrow(X) && error("size of X and y mismatch")
    slice_tocover = collect(1:length(y))

    current_X = @view X[:,:]
    current_y = @view y[:]

    rulelist = ClassificationRule[]
    while true

        best_antecedent = base_beamsearch(current_X, current_y, beam_width)
        # Exit condition
        isnothing(best_antecedent) && break

        covered_indxs = findall(x->x==1,
                            base_interpret(best_antecedent, current_X))
        antecedent_class = mostcommonvalue(current_y[covered_indxs])
        push!(rulelist, convert_solerule(best_antecedent, antecedent_class))

        setdiff!(slice_tocover, slice_tocover[covered_indxs])
        current_X = @view X[slice_tocover, :]
        current_y = @view y[slice_tocover]
    end
    if !allunique(current_y)
        error("Default class can't be created")
    end
    defaultconsequent = current_y[begin]
    return DecisionList(rulelist, defaultconsequent)
end
