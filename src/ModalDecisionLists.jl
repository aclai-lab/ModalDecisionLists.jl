module ModalDecisionLists

# Write your package code here.
using Random

using Reexport
@reexport using SoleBase
@reexport using SoleLogics
@reexport using MultiData
@reexport using SoleModels
@reexport using SoleData

include("core.jl")

export RuleAntecedent

include("myhelp.jl")

export @showlc

module BaseCN2
include("algorithms/base-cn2.jl")
end

module SoleCN2
include("algorithms/sole-cn2.jl")
end

export SoleCN2
export BaseCN2

end
