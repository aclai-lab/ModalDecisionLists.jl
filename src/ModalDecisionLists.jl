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

export SearchMethod
export maptointeger

############################################################################################

include("algorithms/searchmethods/beamsearch.jl")

export BeamSearch
export findbestantecedent

############################################################################################

include("algorithms/searchmethods/randsearch.jl")

export RandSearch

############################################################################################

include("myhelp.jl")
export @showlc

############################################################################################

module BaseCN2
using ModalDecisionLists: RuleAntecedent, SatMask
include("algorithms/base-cn2.jl")
end

############################################################################################

module SoleCN2
using ModalDecisionLists: RuleAntecedent, SatMask
include("algorithms/sole-cn2.jl")
end

############################################################################################

export SoleCN2
export BaseCN2

end # module
