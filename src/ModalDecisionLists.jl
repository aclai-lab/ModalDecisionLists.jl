module ModalDecisionLists

# Write your package code here.
using Random

export BeamSearch, RandSearch

using Reexport
@reexport using SoleBase
@reexport using SoleLogics
@reexport using MultiData
@reexport using SoleModels
@reexport using SoleData

include("core.jl")
# Temporaneamente qui

include("myhelp.jl")
export @showlc

export sequentialcovering

include("algorithms/sequentialcovering.jl")

export BaseCN2

module BaseCN2
using ModalDecisionLists: RuleAntecedent, SatMask
include("algorithms/base-cn2.jl")
end

end
