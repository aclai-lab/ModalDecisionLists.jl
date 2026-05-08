# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractSoleModel

Base abstract type for all symbolic model containers in SoleXplorer.
"""
abstract type AbstractSoleModel end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const XGBoostModel = Union{XGBoostClassifier, XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
# check if dataset or model uses XGBoost
# used to determine if XGBoost-specific setup (watchlist) is needed
has_xgboost_model(ds::DataSet) = has_xgboost_model(ds.mach.model)
has_xgboost_model(model::MLJTuning.EitherTunedModel) = has_xgboost_model(model.model)
has_xgboost_model(::XGBoostModel) = true
has_xgboost_model(::Any) = false

# Check if dataset uses hyperparameter tuning
is_tuned_model(ds::DataSet) = is_tuned_model(ds.mach.model)
is_tuned_model(::MLJTuning.EitherTunedModel) = true
is_tuned_model(::Any) = false

function get_early_stopping_rounds(ds::DataSet)
    return is_tuned_model(ds) ?
        ds.mach.model.model.early_stopping_rounds :
        ds.mach.model.early_stopping_rounds

end

# create XGBoost watchlist for early stopping validation
# throws `ArgumentError` if validation set is empty
function makewatchlist!(ds::DataSet, train::Vector{Int}, valid::Vector{Int})
    isempty(valid) && throw(ArgumentError("No validation data provided, use preprocess valid_ratio parameter"))

    X = get_X(ds)
    y = get_y(ds)
    y_train = @views y[train]
    y_valid = @views y[valid]
    feature_names = String.(propertynames(X))
    if eltype(y) <: CLabel
        y_train = @. MLJ.levelcode(y[train]) - 1 # convert to 0-based indexing
        y_valid = @. MLJ.levelcode(y[valid]) - 1 # convert to 0-based indexing
    end
    dtrain        = XGBoost.DMatrix((@views X[train, :], y_train); feature_names)
    dvalid        = XGBoost.DMatrix((@views X[valid, :], y_valid); feature_names)

    watchlist = XGBoost.OrderedDict(["train" => dtrain, "eval" => dvalid])

    if is_tuned_model(ds)
        ds.mach.model.model.watchlist = watchlist
    else
        ds.mach.model.watchlist = watchlist
    end
end

# configure XGBoost watchlist for fold `i` if early stopping is enabled
function set_watchlist!(ds::DataSet, i::Int)
    # XGBoost supports early stopping. This requires configuring a watchlist and validation ratio
    if get_early_stopping_rounds(ds) > 0
        train = get_train(ds.pidxs[i])
        valid = get_valid(ds.pidxs[i])
        makewatchlist!(ds, train, valid)
    end
end

# ---------------------------------------------------------------------------- #
#                                  solemodel                                   #
# ---------------------------------------------------------------------------- #
# container for collections of symbolic models from cross-validation
mutable struct SoleModel{D} <: AbstractSoleModel
    sole   :: Vector{AbstractModel}

    function SoleModel(::D, sole::Vector{AbstractModel}) where D<:DataSet
        new{D}(sole)
    end
end

# ---------------------------------------------------------------------------- #
#                                  base show                                   #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, solem::SoleModel{D}) where D
    n_models = length(solem.sole)
    dataset_type = D <: DataSet ? string(D) : "Unknown"
    
    print(io, "SoleModel{$dataset_type}")
    print(io, "\n  Number of models: $n_models")
end

function Base.show(io::IO, ::MIME"text/plain", solem::SoleModel{D}) where D
    show(io, solem)
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
# extract the vector of symbolic models from a SoleModel container
solemodels(solem::SoleModel) = solem.sole

# ---------------------------------------------------------------------------- #
#                             internal train_test                              #
# ---------------------------------------------------------------------------- #
# internal cross-validation training implementation
function _train_test(ds::DataSet)::SoleModel
    n_folds   = length(ds.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(ds.pidxs[i]), get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

        has_xgboost_model(ds) && set_watchlist!(ds, i)

        mach = get_mach(ds)
        MLJ.fit!(mach, rows=train, verbosity=0)
        solemodel[i] = apply(mach, X_test, y_test)
    end

    return SoleModel(ds, solemodel)
end

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function train_test(args...; kwargs...)::SoleModel
    ds = setup_dataset(args...; kwargs...)
    _train_test(ds)
end

"""
    train_test(ds::DataSet) -> SoleModel

Direct training interface for pre-configured datasets.
No additional parameters needed.

See also: [`solexplorer`](@ref), [`setup_dataset`](@ref)
"""
train_test(ds::DataSet)::SoleModel = _train_test(ds)
