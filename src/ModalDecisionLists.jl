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


export BaseCN2

module BaseCN2
using ModalDecisionLists: RuleAntecedent, SatMask
include("algorithms/base-cn2.jl")
end

export SoleCN2

module SoleCN2
using ModalDecisionLists: RuleAntecedent, SatMask
include("algorithms/sequentialcovering.jl")
end

############################################################################################

end # module
