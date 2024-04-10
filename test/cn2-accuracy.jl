using Test
using SoleBase: CLabel
using DataFrames
using SoleModels: ClassificationRule, apply, DecisionList, orange_decision_list
using SoleData
using MLJ
using StatsBase
using Random
using RDatasets
using ModalDecisionLists
using ModalDecisionLists: BaseCN2

# SyntheticDatasets.jl


#=
Xy = RDatasets.dataset("mlmRev", "Early")                                   WITH CATEGORICAL
Xy = RDatasets.dataset("mlmRev", "Contraception")                           WITH CATEGORICAL
Xy = RDatasets.dataset("mlmRev", "guPrenat")                                WITH CATEGORICAL
Xy = RDatasets.dataset("survival", "logan")                                 WITH CATEGORICAL

Xy = RDatasets.dataset("psych", "sat.act")
Xy = RDatasets.dataset("gamair", "wesdr")
Xy = RDatasets.dataset("gamair", "wesdr")
Xy = RDatasets.dataset("ISLR", "Weekly")
Xy = RDatasets.dataset("ISLR", "OJ")
Xy = RDatasets.dataset("ISLR", "Caravan")
Xy = RDatasets.dataset("ISLR", "Smarket")
Xy = RDatasets.dataset("ISLR", "Default")
Xy = RDatasets.dataset("Ecdat", "BudgetFood")
Xy = RDatasets.dataset("Ecdat", "Bwages")

=#











# TODO RIGUARDACI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# EfronMorris_dataframe = RDatasets.dataset("pscl", "EfronMorris")
# #=
# 18×7 DataFrame
#  Row │ Name              Team   League  R      Y        N      P
#      │ String            Cat…   Cat…    Int32  Float64  Int32  Float64
# ─────┼─────────────────────────────────────────────────────────────────
#    1 │ Roberto Clemente  Pitts  NL         18    0.4      367    0.346
#    2 │ Frank Robinson    Balt   AL         17    0.378    426    0.298
#    3 │ Frank Howard      Wash   AL         16    0.356    521    0.276
#   ⋮           ⋮            ⋮      ⋮       ⋮       ⋮       ⋮       ⋮
#   17 │ Thurman Munson    NY     AL          8    0.178    408    0.316
#   18 │ Max Alvis         Mil    NL          7    0.156     70    0.2
#                                                         13 rows omitted

# Target:     League
# Exclude:    Name
# =#
# y = EfronMorris_dataframe[:, :League]
# select!(EfronMorris_dataframe, Not([:Name, :League]));

# X_EfronMorris = PropositionalLogiset(EfronMorris_dataframe)
