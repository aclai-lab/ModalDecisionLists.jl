using ModalDecisionLists
using SoleModels: DecisionList, DecisionEnsemble
using SoleModels: default_weights
using SoleBase: CLabel
using SoleData: AbstractLogiset


const RandomDecisionLists{O, A, W} = DecisionEnsemble{O, <:DecisionList, A, W} 

function RandomDecisionLists(
    lists::Vector{<:DecisionList}, 
    aggregation::Union{Nothing, Base.Callable},
    info::NamedTuple = (;),
)
    return DecisionEnsemble(
            lists, 
            aggregation, 
            nothing, # weights
            info        # TODO: merge with (type="RandomDecisionList")?
        )
end


lists(m::RandomDecisionLists) = models(m)
nlists(m::RandomDecisionLists) = length(lists(m))



function build_rdl(
    X::AbstractLogiset,
    y::AbstractVector{<:CLabel},
    poslabel::CLabel,
    num_models::Integer,
    w::Union{Nothing, AbstractVector{U}, Symbol} = default_weights(length(y));
    samples_ratio_per_model::Real = 0.8,
    aggregation_function::Union{Nothing, Base.Callable} = nothing,
    kwargs...
)::DecisionEnsemble where {U<:Real}
    @assert (0.0 < samples_ratio_per_model < 1.0) "Parameter `samples_ratio_per_model` must be in [0, 1]"
    @assert (num_models > 0) "Parameter `num_models` must be ≥ 1"

    num_samples = ninstances(X)
    n_samples_per_model = round(Integer, ninstances(X) * samples_ratio_per_model)

    models = DecisionList[]

    # TODO: parallelization?
    for i = 1 : num_models
        # Extract 'n_samples_per_model' random integers in [1, num_samples]
        permutated_indices = randperm(num_samples)
        model_dataset_indices = permutated_indices[1:n_samples_per_model] 

        # use those indices to extract a dataset from X
        X_model = slicedataset(X, model_dataset_indices)
        y_model = @view y[model_dataset_indices]
        w_model = (w isa AbstractVector) ? @view(w[model_dataset_indices]) : w      # w might be nothing

        model = irepstar(X_model, y_model, poslabel, w_model; kwargs...) 
        push!(models, model)
    end

    return RandomDecisionLists(models, aggregation_function)
end