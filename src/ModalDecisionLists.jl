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

include("loss-functions.jl")

using .LossFunctions

include("search.jl")

export sequentialcovering

include("algorithms/sequentialcovering.jl")
# include("algorithms/sequentialcovering-unordered.jl")

export irepstar
export initialize_antecedents
export ripperk

module BaseCN2
using ModalDecisionLists: SatMask
include("algorithms/base-cn2.jl")
end

export ExtendedSequentialCovering
export OrderedCN2Learner
# export build_cn2

# MLJ Interface
include("interfaces/MLJ.jl")


include("deprecate.jl")

using .MLJInterface

end
