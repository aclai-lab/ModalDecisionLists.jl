module ModalDecisionLists

# Write your package code here.
using Random

export BeamSearch, RandSearch, AtomSearch, SearchMethod

using Reexport
@reexport using SoleBase
@reexport using SoleLogics
@reexport using MultiData
@reexport using SoleModels
@reexport using SoleData


include("measures.jl")

using .Measures

include("core.jl")
# Temporaneamente qui

export sequentialcovering

include("algorithms/sequentialcovering.jl")


module BaseCN2
using ModalDecisionLists: SatMask
include("algorithms/base-cn2.jl")
end

export SequentialCoveringLearner
export OrderedCN2Learner
export build_cn2

# Interface
include("interfaces/MLJ.jl")

using .MLJInterface

end
