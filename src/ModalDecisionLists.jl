module ModalDecisionLists

# Write your package code here.
using Random

export BeamSearch, RandSearch, SearchMethod

using Reexport
@reexport using SoleBase
@reexport using SoleLogics
@reexport using MultiData
@reexport using SoleData
@reexport using SoleModels

include("loss-functions.jl")

using .LossFunctions

include("core.jl")

export sequentialcovering

include("algorithms/sequentialcovering.jl")


module BaseCN2
using ModalDecisionLists: RuleAntecedent, SatMask
include("algorithms/base-cn2.jl")
end

export SequentialCoveringLearner
export build_cn2

# MLJ Interface
include("interfaces/MLJ.jl")


include("deprecate.jl")

using .MLJInterface

end
