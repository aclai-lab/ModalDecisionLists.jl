module ModalDecisionLists

using Random

export BeamSearch, RandSearch, AtomSearch, SearchMethod

using Reexport
@reexport using SoleBase
@reexport using SoleLogics
@reexport using SoleData
@reexport using SoleModels

include("loss-functions.jl")

using .LossFunctions

include("core.jl")

export sequentialcovering

include("algorithms/sequentialcovering.jl")
# include("algorithms/sequentialcovering-unordered.jl")


module BaseCN2
using ModalDecisionLists: SatMask
include("algorithms/base-cn2.jl")
end

export ExtendedSequentialCovering
export OrderedCN2Learner
export build_cn2

# MLJ Interface
include("interfaces/MLJ.jl")


include("deprecate.jl")

using .MLJInterface

end
