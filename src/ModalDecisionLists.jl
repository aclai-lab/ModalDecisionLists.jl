module ModalDecisionLists

using Random

export BeamSearch, RandSearch, SearchMethod

using Reexport
@reexport using SoleBase
@reexport using SoleLogics
@reexport using SoleData
@reexport using SoleModels




include("utils.jl")

include("metrics.jl")

using .Metrics

include("core.jl")


include("featureselection/default-selector.jl")
include("featureselection/weighted-random-selector.jl")

export DefaultFeatureSelector, WeightedRandomFeatureSelector

include("loss-functions.jl")

using .LossFunctions

include("search.jl")


export AtomGenerator
export sequentialcovering

include("algorithms/sequentialcovering.jl")
# include("algorithms/sequentialcovering-unordered.jl")

export irepstar
export ripperk

include("ensemble_learning.jl")
export build_ensemble
export atoms, natoms

include("random-lists.jl")
export RandomDecisionListEnsemble
export build_random_lists


module BaseCN2
using ModalDecisionLists: SatMask
include("algorithms/base-cn2.jl")
end

include("interfaces/MLJ.jl")
export ExtendedSequentialCovering, OrderedCN2Learner
export DecisionListClassifier, BaggedEnsembleClassifier
export RipperListClassifier, RandomDecisionListEnsembleClassifier


include("deprecate.jl")

using .MLJInterface

end
