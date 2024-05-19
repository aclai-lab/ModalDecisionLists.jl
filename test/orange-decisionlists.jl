using SoleData
using ModalDecisionLists
using SoleModels: ClassificationRule, apply, DecisionList, bestguess, orange_decision_list
using ModalDecisionLists: preprocess_inputdata
using ModalDecisionLists.Measures: laplace_accuracy
using CategoricalArrays: CategoricalValue, CategoricalArray
using RDatasets
using MLJ
using CSV
#
# Decision list ottenuta con CN2-Orange dalle prime 1000 istamze del dataset Yeast
#
#    beam-width       : 3
#    discretizedomain : true
#
abalone1000_dl_orange = """
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0075 AND Sex!=I THEN Rings=3 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.02 AND Diameter>=0.11 THEN Rings=4 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.035 AND Shucked_weight>=0.0095 THEN Rings=4 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0075 AND Height>=0.05 THEN Rings=4 -0.0
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.035 AND Length<=0.11 AND Diameter>=0.09 THEN Rings=3 -0.0
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.035 AND Length<=0.13 AND Diameter>=0.1 THEN Rings=3 -0.0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0045 AND Diameter>=0.1 THEN Rings=2 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.048 AND Sex==F AND Diameter>=0.23 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Sex==F AND Diameter>=0.275 THEN Rings=8 -0.0
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.185 AND Height<=0.035 AND Viscera_weight>=0.005 THEN Rings=3 -0.0
    [1, 0, 2, 10, 14, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Diameter<=0.185 THEN Rings=5 -2.022589457504889
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Sex==F AND Diameter>=0.25 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.345 AND Sex==F AND Length<=0.325 AND Diameter>=0.26 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.345 AND Sex==F AND Diameter>=0.26 THEN Rings=10 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.1915 AND Sex==F AND Diameter>=0.27 THEN Rings=5 -0.0
    [0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.049 AND Length<=0.25 THEN Rings=4 -0.9182958340544896
    [0, 0, 0, 0, 1, 4, 14, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Whole_weight>=0.169 THEN Rings=7 -1.8640054628542204
    [0, 0, 0, 0, 6, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.168 AND Whole_weight<=0.0975 THEN Rings=5 -1.7381493331928664
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.34 AND Length>=0.33 AND Whole_weight>=0.222 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.049 AND Length>=0.32 AND Viscera_weight>=0.038 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.075 AND Viscera_weight>=0.061 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.285 AND Sex==F AND Diameter>=0.28 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.08 AND Height>=0.105 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.08 AND Sex==F AND Diameter>=0.295 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Sex==F AND Diameter>=0.29 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.295 AND Sex==F AND Length<=0.38 AND Diameter>=0.29 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.295 AND Sex==F AND Diameter>=0.29 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 15, 16, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.08 AND Shucked_weight>=0.0705 THEN Rings=7 -1.9282017867125283
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.05 AND Whole_weight>=0.1575 AND Diameter>=0.245 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.168 AND Viscera_weight>=0.041 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.1775 AND Viscera_weight>=0.041 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.265 AND Viscera_weight>=0.041 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.27 AND Shucked_weight>=0.0645 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.35 AND Viscera_weight>=0.041 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.355 AND Viscera_weight>=0.041 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.045 AND Length>=0.395 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.199 AND Shucked_weight>=0.0645 AND Diameter>=0.28 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0475 AND Shucked_weight>=0.104 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.075 AND Shucked_weight>=0.0645 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.081 AND Length>=0.37 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.08 AND Length>=0.38 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0885 AND Whole_weight>=0.2255 AND Sex!=M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.22 AND Viscera_weight>=0.041 AND Length>=0.36 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.22 AND Viscera_weight>=0.041 AND Length>=0.315 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0525 AND Viscera_weight>=0.0525 AND Sex!=F THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.36 AND Height>=0.105 AND Whole_weight>=0.2765 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Height>=0.105 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length>=0.355 AND Sex!=M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0525 AND Length>=0.33 AND Length>=0.385 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length<=0.275 AND Sex!=I THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length<=0.275 AND Diameter>=0.215 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length<=0.275 AND Whole_weight>=0.1105 THEN Rings=6 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length<=0.275 AND Diameter>=0.205 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Length>=0.33 AND Sex!=I THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0525 AND Length>=0.33 AND Sex==M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Shucked_weight>=0.0545 AND Shucked_weight>=0.065 AND Diameter>=0.245 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Shucked_weight>=0.0545 AND Diameter>=0.255 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length>=0.325 AND Sex!=M THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.375 AND Length>=0.325 AND Diameter>=0.24 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.275 AND Length>=0.325 AND Diameter>=0.275 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.24 AND Length>=0.325 AND Diameter>=0.3 THEN Rings=9 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0525 AND Length>=0.325 AND Sex!=F THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.252 AND Length>=0.32 AND Diameter>=0.305 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.2605 AND Length>=0.405 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.265 AND Length>=0.39 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.281 AND Diameter>=0.305 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.38 AND Length>=0.38 THEN Rings=9 -1.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.2845 AND Diameter>=0.31 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.286 AND Diameter>=0.315 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.2885 AND Length>=0.42 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.108 AND Length>=0.4 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0565 AND Shucked_weight>=0.0545 AND Length>=0.4 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.057 AND Length>=0.32 AND Diameter>=0.31 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.058 AND Length>=0.405 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0595 AND Length>=0.395 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.116 AND Length>=0.39 AND Length<=0.4 AND Diameter>=0.31 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex==F AND Length>=0.435 AND Diameter>=0.35 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex==F AND Length>=0.435 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex==F AND Whole_weight>=0.363 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex==F AND Diameter>=0.32 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.067 AND Sex==F AND Height>=0.11 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex==F AND Height>=0.13 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.067 AND Height>=0.105 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.067 AND Height>=0.11 AND Shucked_weight>=0.1815 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0685 AND Height>=0.105 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0695 AND Height>=0.11 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.325 AND Sex==M AND Height>=0.14 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Height>=0.125 THEN Rings=9 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Diameter<=0.21 AND Height>=0.1 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Shell_weight>=0.115 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Length>=0.4 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Viscera_weight<=0.0335 AND Height>=0.08 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Length>=0.31 AND Diameter>=0.225 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M AND Height>=0.085 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex!=I AND Sex!=M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Length>=0.43 AND Viscera_weight>=0.103 AND Sex!=F THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Length>=0.43 AND Whole_weight>=0.4525 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Length>=0.43 AND Sex!=I THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.402 AND Sex==M AND Shucked_weight>=0.171 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.402 AND Sex==M AND Height>=0.135 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==F AND Length>=0.425 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex!=I AND Diameter<=0.34 AND Height>=0.13 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex!=I AND Diameter>=0.36 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.115 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.425 AND Sex==F AND Whole_weight>=0.38 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex==F AND Diameter>=0.325 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.425 AND Sex==F AND Diameter>=0.345 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.402 AND Sex==M AND Length>=0.445 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.12 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.402 AND Sex==M AND Height>=0.11 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.402 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.406 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.12 AND Shucked_weight>=0.155 AND Sex!=I THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.33 AND Diameter>=0.325 AND Height>=0.12 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.335 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.335 AND Sex==F AND Height>=0.12 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.345 AND Sex==M AND Diameter>=0.345 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0785 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.34 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0785 AND Whole_weight>=0.377 AND Length>=0.495 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4085 AND Shucked_weight>=0.155 AND Sex!=I THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.34 AND Sex==F AND Height>=0.14 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.34 AND Sex==F AND Height>=0.13 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.13 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4085 AND Whole_weight>=0.382 AND Sex!=F THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.34 AND Sex==F AND Whole_weight>=0.451 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.34 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.3795 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.3795 AND Shell_weight>=0.12 AND Diameter>=0.345 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.425 AND Diameter>=0.325 AND Whole_weight>=0.425 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.34 AND Shucked_weight>=0.1535 AND Length>=0.43 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.43 AND Diameter>=0.325 AND Whole_weight>=0.3595 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.43 AND Diameter>=0.325 AND Height>=0.12 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.435 AND Diameter>=0.325 AND Height>=0.11 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.435 AND Diameter>=0.325 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4085 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.415 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4205 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.421 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.431 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.433 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.08 AND Length>=0.45 AND Diameter>=0.38 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0805 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.084 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.435 AND Length>=0.44 AND Diameter>=0.355 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.45 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.45 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.452 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.458 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.095 AND Sex==F AND Diameter>=0.375 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.135 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.435 AND Length>=0.385 AND Viscera_weight>=0.092 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.435 AND Shell_weight>=0.045 AND Height>=0.11 AND Length>=0.405 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0935 AND Sex!=I AND Sex!=M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0935 AND Sex!=I AND Height>=0.15 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.435 AND Shell_weight>=0.045 AND Diameter>=0.32 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.44 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.135 AND Sex==F AND Length>=0.485 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.095 AND Height>=0.135 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.099 AND Sex!=I AND Diameter>=0.375 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.099 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4705 AND Sex!=I THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.458 AND Shucked_weight>=0.1215 AND Height>=0.135 AND Diameter>=0.35 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.445 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.458 AND Shucked_weight>=0.1215 AND Whole_weight>=0.458 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4705 AND Length>=0.445 AND Diameter>=0.37 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.135 AND Sex==F AND Diameter<=0.36 AND Diameter>=0.36 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.135 AND Sex==F AND Height>=0.14 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.14 AND Sex!=I AND Sex!=M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.14 AND Height>=0.135 AND Diameter<=0.35 AND Height>=0.14 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.471 AND Length>=0.385 AND Viscera_weight>=0.099 AND Diameter>=0.375 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.472 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.482 AND Sex!=I AND Sex!=M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.482 AND Sex!=I AND Diameter>=0.38 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4835 AND Sex!=I AND Diameter>=0.355 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4835 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.487 AND Viscera_weight>=0.0855 AND Whole_weight>=0.487 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.488 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4885 AND Length>=0.445 AND Height>=0.135 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4885 AND Shell_weight>=0.145 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.498 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1 AND Length>=0.385 AND Height>=0.145 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.107 AND Sex==F AND Height>=0.135 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.107 AND Sex==M AND Diameter>=0.375 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.101 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.107 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1075 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.108 AND Shell_weight>=0.22 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.108 AND Sex!=I AND Diameter>=0.385 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.108 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1125 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1125 AND Sex==F AND Diameter>=0.405 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Viscera_weight<=0.1135 AND Sex==F AND Diameter>=0.395 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1135 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1135 AND Shell_weight>=0.1465 AND Length>=0.52 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1135 AND Shell_weight>=0.1465 AND Diameter>=0.39 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.115 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1155 AND Shell_weight>=0.17 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1065 AND Length>=0.385 AND Shucked_weight>=0.2835 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1065 AND Length>=0.385 AND Length>=0.47 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.116 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.116 AND Length>=0.385 AND Viscera_weight>=0.116 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.4985 AND Height>=0.14 AND Diameter>=0.37 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.514 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.35 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.35 AND Sex==F AND Whole_weight>=0.5425 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1165 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1185 AND Sex==M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1165 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.14 AND Sex!=I AND Diameter<=0.365 AND Diameter>=0.365 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.14 AND Sex!=I AND Diameter>=0.37 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.14 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1495 AND Sex!=I AND Diameter>=0.4 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1495 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.15 AND Sex!=I AND Diameter>=0.395 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.155 AND Sex==M AND Shucked_weight>=0.3195 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.15 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.536 AND Sex==M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shucked_weight<=0.185 AND Sex==F AND Length<=0.49 AND Diameter>=0.365 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.185 AND Sex==F AND Diameter>=0.375 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.187 AND Sex==F AND Diameter<=0.36 THEN Rings=15 -1.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.187 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.189 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1925 AND Sex==F AND Diameter>=0.385 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1925 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.5465 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1975 AND Sex==F AND Diameter>=0.39 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1995 AND Sex==F AND Diameter>=0.38 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1995 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2 AND Length>=0.44 AND Viscera_weight>=0.145 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2005 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.204 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.211 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2125 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2135 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.215 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2155 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.216 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Length<=0.47 AND Viscera_weight>=0.1915 THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.47 AND Sex==M AND Diameter>=0.38 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.47 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.155 AND Sex==M AND Diameter>=0.41 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.155 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.155 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.5615 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.115 AND Viscera_weight>=0.1845 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.115 AND Whole_weight>=0.669 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.16 AND Sex==M AND Diameter>=0.39 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.5915 AND Sex==M AND Diameter>=0.4 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Whole_weight<=0.5915 AND Sex==M THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.216 AND Shucked_weight>=0.1215 AND Whole_weight>=0.521 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2075 AND Shucked_weight>=0.1215 AND Whole_weight>=0.514 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.216 AND Shucked_weight>=0.1215 AND Height>=0.14 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.568 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2165 AND Shucked_weight>=0.1215 AND Diameter>=0.38 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Shell_weight>=0.24 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Shell_weight>=0.225 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Shell_weight>=0.2 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Sex==M THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Sex==F AND Diameter>=0.38 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Length>=0.475 AND Height>=0.14 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.156 AND Length>=0.385 AND Length>=0.475 AND Height>=0.15 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.16 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.16 AND Length>=0.385 AND Height>=0.15 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1635 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.165 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.166 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.168 AND Length>=0.385 AND Length>=0.475 AND Diameter>=0.365 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.5735 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1405 AND Shucked_weight>=0.1215 AND Length>=0.46 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1405 AND Shucked_weight>=0.1215 AND Whole_weight>=0.547 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1465 AND Length>=0.385 AND Diameter>=0.375 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.227 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2305 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Length>=0.475 AND Sex!=F THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Sex==F AND Diameter>=0.415 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Height>=0.11 AND Whole_weight>=0.581 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.375 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.604 AND Height>=0.135 AND Diameter>=0.39 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.131 AND Sex==F AND Height>=0.19 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.604 AND Sex==F AND Diameter>=0.4 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Viscera_weight<=0.129 AND Sex==F AND Diameter>=0.455 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.604 AND Sex==F AND Height>=0.15 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1695 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.604 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.609 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.627 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.627 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.63 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6415 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6435 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Viscera_weight<=0.13 AND Sex==M AND Diameter>=0.42 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.131 AND Sex==M AND Diameter>=0.405 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1315 AND Sex==M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6435 AND Sex==F AND Diameter>=0.405 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.644 AND Sex==F AND Diameter>=0.41 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6445 AND Sex==F AND Diameter>=0.4 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.646 AND Sex==F AND Diameter>=0.405 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1315 AND Sex==F AND Diameter>=0.41 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1315 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.133 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Sex!=I AND Diameter>=0.435 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.137 AND Sex!=I AND Diameter>=0.41 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.138 AND Sex!=I AND Diameter>=0.415 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1385 AND Sex==M AND Diameter>=0.435 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1385 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1385 AND Sex!=I THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.139 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1395 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.655 AND Sex==F AND Diameter>=0.39 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.655 AND Sex==M AND Diameter>=0.405 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6515 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.655 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6565 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2385 AND Sex==F AND Diameter>=0.44 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.24 AND Sex==F AND Height>=0.165 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2405 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.38 AND Height>=0.175 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.38 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.49 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Length<=0.495 AND Sex!=I THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1415 AND Sex==F AND Height>=0.15 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.244 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2505 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.254 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.255 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1425 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.39 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.39 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.495 AND Height>=0.11 AND Diameter>=0.39 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2575 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.5 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.5 AND Sex==F AND Whole_weight>=0.7155 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1425 AND Sex==F AND Diameter>=0.42 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1425 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1435 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1455 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.146 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2635 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.264 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1495 AND Sex==F AND Diameter>=0.45 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1495 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.151 AND Sex!=I THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.152 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.152 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2665 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2665 AND Sex==F AND Shucked_weight>=0.2665 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2705 AND Sex==F AND Diameter>=0.43 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2705 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.445 AND Shell_weight>=0.045 AND Diameter>=0.365 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.194 AND Sex==F AND Diameter>=0.415 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.194 AND Sex==M AND Diameter>=0.435 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1525 AND Sex==M AND Diameter>=0.44 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1525 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1525 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.155 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1555 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.659 AND Shucked_weight>=0.1215 AND Shucked_weight>=0.2705 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2725 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2735 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.275 AND Sex==M THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.275 AND Sex==F AND Diameter>=0.425 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2755 AND Sex==F AND Height>=0.17 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.276 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2775 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2785 AND Sex==F AND Height>=0.16 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2785 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.279 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2795 AND Sex==F AND Diameter>=0.415 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2795 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6805 AND Shucked_weight>=0.1215 AND Height>=0.17 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6965 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2805 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.17 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.699 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.185 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.185 AND Shucked_weight>=0.1215 AND Shucked_weight>=0.225 AND Diameter>=0.37 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.185 AND Shucked_weight>=0.1215 AND Shucked_weight>=0.25 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1955 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.51 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.2 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.2 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.202 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.202 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.51 AND Shucked_weight>=0.1215 AND Diameter>=0.4 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.282 AND Sex==F AND Diameter>=0.445 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.282 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2825 AND Sex!=I THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.289 AND Sex==M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.289 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.291 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.292 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.515 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.724 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.515 AND Sex==F AND Whole_weight>=0.834 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.515 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.781 AND Sex==F AND Height>=0.145 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.781 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Sex==F AND Diameter>=0.43 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.295 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.298 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2985 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Whole_weight<=0.752 AND Shucked_weight>=0.1215 AND Diameter>=0.42 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.781 AND Shucked_weight>=0.1215 AND Diameter>=0.45 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.791 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.795 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.52 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.52 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.156 AND Sex==M THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.156 AND Sex!=I THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.157 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1575 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1585 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.159 AND Sex!=I THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.4 AND Sex==M THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.4 AND Sex==F AND Whole_weight>=0.8215 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.4 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.795 AND Shucked_weight>=0.1215 AND Diameter>=0.405 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.797 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.405 AND Sex!=I AND Height>=0.185 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.405 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8 AND Shucked_weight>=0.1215 AND Diameter>=0.44 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.805 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8055 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.806 AND Shucked_weight>=0.1215 AND Diameter>=0.425 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8075 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8105 AND Sex==F AND Diameter>=0.435 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8115 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.826 AND Sex==F AND Diameter>=0.445 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8285 AND Sex==F AND Height>=0.145 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.835 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.836 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8375 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8425 AND Sex==F AND Diameter>=0.44 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.843 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8445 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.849 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8515 AND Sex==F AND Diameter>=0.43 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8515 AND Sex==F AND Height>=0.16 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8515 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1615 AND Sex==F AND Height>=0.185 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1615 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.162 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Viscera_weight<=0.1635 AND Sex==F AND Diameter>=0.45 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1635 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.164 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1655 AND Sex!=I AND Diameter>=0.465 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1655 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1665 AND Sex!=I THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.167 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.525 AND Sex!=I AND Height>=0.155 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.525 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.525 AND Shucked_weight>=0.1215 AND Diameter>=0.41 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.41 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8525 AND Shucked_weight>=0.1215 AND Diameter>=0.455 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8565 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8645 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.865 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8735 AND Shucked_weight>=0.1215 AND Diameter>=0.445 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.305 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.309 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.31 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.313 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3135 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3145 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.315 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3155 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.319 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.322 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.415 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.874 AND Sex!=I AND Diameter>=0.45 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3275 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3365 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Diameter<=0.415 AND Shucked_weight>=0.1215 AND Diameter>=0.415 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.53 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.535 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.23 AND Sex!=I AND Sex!=M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.535 AND Shucked_weight>=0.1215 AND Diameter>=0.42 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.42 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1695 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1695 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1705 AND Sex!=I THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1735 AND Sex==M THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.875 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.2375 AND Sex==M AND Whole_weight>=0.93 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.2375 AND Sex==M THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.2375 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.245 AND Sex!=I AND Sex!=M THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.245 AND Sex!=I AND Whole_weight>=1.0635 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.245 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.25 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.25 AND Sex==F AND Diameter>=0.46 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.25 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.178 AND Sex==F AND Height>=0.15 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.178 AND Sex==M AND Whole_weight>=1.07 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.178 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.181 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.182 AND Sex==M THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.183 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.875 AND Shucked_weight>=0.1215 AND Diameter>=0.425 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.883 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.8865 AND Shucked_weight>=0.1215 AND Diameter>=0.445 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9055 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9075 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.909 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.255 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.425 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.425 AND Shucked_weight>=0.1215 AND Diameter>=0.425 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.915 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.915 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.92 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.922 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.928 AND Sex==F AND Diameter>=0.475 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9285 AND Sex==F AND Diameter>=0.44 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.931 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9365 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.54 AND Sex==M THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.54 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.55 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.435 AND Sex==F AND Diameter>=0.435 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.939 AND Sex==F AND Diameter>=0.46 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.939 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.555 AND Sex!=I AND Height>=0.16 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.555 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.26 AND Sex==F AND Diameter>=0.46 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.26 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.19 AND Sex!=I AND Sex!=M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.939 AND Shucked_weight>=0.1215 AND Diameter>=0.47 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.339 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.342 AND Sex!=I THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3445 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.347 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3605 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9395 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9395 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.945 AND Sex!=I AND Sex!=M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.945 AND Sex!=I AND Height>=0.14 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.945 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.366 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9535 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9535 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.367 AND Sex==M THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.367 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.44 AND Sex!=I AND Height>=0.155 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.44 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.445 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  IF Viscera_weight<=0.19 AND Sex!=I AND Diameter>=0.495 THEN Rings=26 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.19 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.192 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9665 AND Shucked_weight>=0.1215 AND Diameter>=0.475 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.97 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.987 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Sex!=I THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shucked_weight>=0.1215 AND Diameter>=0.49 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9915 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9935 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.994 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9955 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.997 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9975 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0035 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0075 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0105 AND Sex!=I THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.011 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.56 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0115 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.012 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.013 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.016 AND Sex!=I THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0165 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0225 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.029 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Whole_weight<=1.036 AND Sex!=I THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0385 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.197 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.565 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.565 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.203 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0525 AND Sex==F AND Diameter>=0.48 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.053 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.054 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.054 AND Shucked_weight>=0.1215 AND Diameter>=0.495 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.055 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.056 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0565 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0595 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0605 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Whole_weight<=1.066 AND Sex!=I THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0735 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3705 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3755 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.275 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0875 AND Sex!=I AND Height>=0.19 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0915 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.0915 AND Sex!=I THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.094 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.098 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.098 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1015 AND Sex==F AND Diameter>=0.475 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1015 AND Sex!=I THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.102 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1025 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1045 AND Sex!=I AND Diameter>=0.5 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.105 AND Sex!=I AND Diameter>=0.47 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1095 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1095 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Shucked_weight<=0.378 AND Sex==F AND Whole_weight>=1.1835 THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.378 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Shucked_weight<=0.381 AND Sex!=I THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3875 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4015 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.403 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shucked_weight<=0.408 AND Sex!=I THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.412 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.416 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4205 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4245 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.428 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4305 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.57 AND Diameter>=0.48 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.57 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.14 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Shucked_weight<=0.4385 AND Sex==M THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.14 AND Sex==F AND Diameter>=0.475 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.14 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.145 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4385 AND Sex==F AND Diameter>=0.505 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4385 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Sex==F AND Height>=0.15 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.45 AND Sex==M THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.455 AND Sex==M THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.465 AND Sex==M AND Whole_weight>=1.3415 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.465 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.47 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.28 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.439 AND Shucked_weight>=0.1215 AND Diameter>=0.53 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4445 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4525 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Shucked_weight<=0.4535 AND Sex!=I THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.458 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4585 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.58 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.58 AND Sex!=I THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.58 AND Shucked_weight>=0.1215 AND Diameter>=0.49 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4635 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shucked_weight<=0.4635 AND Sex!=I THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4675 AND Shucked_weight>=0.1215 AND Diameter>=0.485 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.469 AND Sex==F AND Height>=0.19 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.471 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4775 AND Sex==F AND Diameter>=0.51 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Height>=0.16 AND Height>=0.18 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Diameter>=0.515 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.212 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2165 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2185 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.225 AND Sex!=I THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.48 AND Sex==M THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4835 AND Sex==M THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.485 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.486 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.488 AND Sex==M THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.493 AND Sex==F AND Diameter>=0.53 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4935 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.4955 AND Sex==M THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Shucked_weight<=0.496 AND Sex==M THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.498 AND Sex==F AND Diameter>=0.5 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.498 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1615 AND Sex==F AND Diameter>=0.48 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.186 AND Sex==F AND Diameter>=0.5 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.199 AND Sex==F AND Diameter>=0.48 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5015 AND Sex!=I THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5065 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.507 AND Sex!=I THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.509 AND Sex!=I THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5095 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.51 AND Sex!=I THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5125 AND Sex!=I THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shucked_weight<=0.515 AND Sex!=I THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.518 AND Sex!=I THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.218 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2325 AND Sex==F AND Height>=0.175 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2465 AND Sex==F AND Diameter>=0.495 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.247 AND Sex==F AND Diameter>=0.49 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2475 AND Sex==F AND Diameter>=0.535 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2575 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2575 AND Sex==F AND Diameter>=0.485 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Shucked_weight>=0.225 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Length>=0.445 AND Height>=0.13 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Diameter>=0.355 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Whole_weight>=0.377 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Diameter>=0.325 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.1215 AND Shell_weight>=0.12 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shell_weight>=0.045 AND Height>=0.11 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Length>=0.385 AND Diameter>=0.31 THEN Rings=6 -0.8112781244591328
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Length>=0.385 AND Length>=0.395 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Diameter>=0.295 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.056 AND Height>=0.095 THEN Rings=6 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.0585 AND Height>=0.08 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Whole_weight>=0.1485 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Height>=0.085 AND Diameter>=0.225 THEN Rings=7 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shucked_weight>=0.06 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.6 AND Sex==M THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.61 AND Sex==F AND Whole_weight>=1.2095 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.261 AND Sex==M THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2615 AND Sex==M THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2415 AND Sex==F AND Diameter>=0.515 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2445 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.61 AND Sex==M THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.49 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.49 AND Length>=0.625 AND Diameter>=0.49 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.519 AND Sex==M THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.519 AND Sex!=I THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.526 AND Sex!=I THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2605 AND Length>=0.615 AND Diameter>=0.52 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5295 AND Sex!=I THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2895 AND Sex==F AND Diameter>=0.51 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.309 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.615 AND Sex==M THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.615 AND Sex==F AND Diameter>=0.5 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2915 AND Sex==F AND Diameter>=0.55 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.309 AND Length>=0.635 AND Diameter>=0.585 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Whole_weight<=1.3135 AND Length>=0.63 AND Diameter>=0.5 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.32 AND Sex==M THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2635 AND Sex==M THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Viscera_weight<=0.2665 AND Sex==M THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2735 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.275 AND Length>=0.62 AND Diameter>=0.505 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2945 AND Sex==F AND Diameter>=0.485 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.62 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.35 AND Sex==M THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2805 AND Sex==F AND Diameter>=0.56 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2895 AND Sex==F AND Diameter>=0.545 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.297 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.297 AND Sex==F AND Diameter>=0.505 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.298 AND Sex==M THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.3905 AND Sex==M THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.5 AND Length>=0.63 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.63 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.3905 AND Sex==F AND Diameter>=0.55 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.41 AND Sex==F AND Diameter>=0.535 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Whole_weight<=1.4225 AND Sex==F AND Diameter>=0.525 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.55 AND Sex==M THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5755 AND Sex==M THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.578 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.581 AND Sex==F AND Diameter>=0.555 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.585 AND Sex==F AND Diameter>=0.58 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.395 AND Sex==M THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==M AND Length>=0.73 AND Whole_weight>=2.499 THEN Rings=17 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Diameter>=0.225 AND Diameter>=0.235 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Diameter>=0.225 AND Length>=0.3 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Whole_weight>=0.124 AND Height>=0.085 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Whole_weight>=0.124 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.5975 AND Sex==M THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shell_weight>=0.035 AND Diameter>=0.225 THEN Rings=8 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Shell_weight>=0.035 THEN Rings=5 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Diameter>=0.21 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I AND Height>=0.07 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==I THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.566 AND Sex!=F THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.815 AND Length>=0.705 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.7595 AND Sex!=F THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.7595 AND Shucked_weight>=0.833 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Sex!=F AND Height>=0.225 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex!=F AND Length>=0.725 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Sex!=F AND Diameter>=0.58 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Sex!=F AND Shucked_weight>=0.7425 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex!=F THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.2 AND Length>=0.725 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Height>=0.2 AND Shell_weight>=0.65 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.2 AND Height>=0.21 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.475 AND Height>=0.2 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shell_weight>=0.475 AND Shucked_weight>=0.765 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.545 AND Shucked_weight>=0.815 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.4635 AND Diameter>=0.525 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.566 AND Length>=0.675 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.545 AND Viscera_weight>=0.3265 AND Diameter>=0.565 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  IF Diameter>=0.545 AND Diameter>=0.585 THEN Rings=29 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Diameter>=0.545 AND Height>=0.195 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.545 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.455 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.66 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.64 AND Diameter>=0.51 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.645 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Sex==F THEN Rings=6 -0.0
    [1, 1, 7, 18, 31, 60, 97, 84, 109, 117, 91, 79, 71, 49, 52, 29, 28, 18, 17, 16, 9, 6, 5, 1, 1]  IF TRUE THEN Rings=10 -3.984037499126841
""" |> orange_decision_list

abalone1000_no_categorical_dl_orange = """
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0105 AND Shell_weight>=0.0155 AND Diameter>=0.175 THEN Rings=5 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.02 AND Diameter>=0.11 THEN Rings=4 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.035 AND Shucked_weight>=0.0095 THEN Rings=4 -0.0
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.035 AND Length<=0.11 AND Diameter>=0.09 THEN Rings=3 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0075 AND Diameter>=0.11 AND Shucked_weight>=0.0075 THEN Rings=5 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.04 AND Length>=0.235 THEN Rings=5 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.045 AND Viscera_weight>=0.0235 THEN Rings=5 -0.0
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.045 AND Viscera_weight>=0.006 THEN Rings=4 -0.0
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.045 AND Viscera_weight<=0.003 AND Length>=0.13 THEN Rings=3 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Height<=0.045 AND Length>=0.205 THEN Rings=7 -0.0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.018 AND Length<=0.155 AND Length<=0.15 AND Diameter>=0.1 THEN Rings=2 -0.0
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.018 AND Length<=0.17 AND Height>=0.05 THEN Rings=4 -0.0
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Shell_weight<=0.018 AND Diameter<=0.125 AND Viscera_weight>=0.005 THEN Rings=3 -0.0
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Diameter<=0.195 AND Shucked_weight>=0.042 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.04 AND Length>=0.29 AND Whole_weight>=0.101 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.04 AND Length<=0.205 AND Whole_weight>=0.046 THEN Rings=6 -0.0
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.018 AND Whole_weight>=0.0665 AND Height>=0.065 THEN Rings=3 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.17 AND Length<=0.175 AND Diameter>=0.125 THEN Rings=5 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Diameter<=0.17 AND Length<=0.195 AND Diameter>=0.145 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Diameter<=0.17 AND Diameter>=0.17 THEN Rings=7 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.04 AND Diameter>=0.165 AND Height>=0.085 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.04 AND Shell_weight>=0.021 AND Shucked_weight>=0.04 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0405 AND Shell_weight>=0.021 AND Diameter>=0.225 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.043 AND Whole_weight>=0.1085 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.0435 AND Shell_weight>=0.021 AND Length>=0.29 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.045 AND Shell_weight>=0.021 AND Diameter<=0.185 AND Height>=0.07 THEN Rings=6 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.045 AND Shell_weight>=0.021 AND Diameter<=0.19 AND Whole_weight>=0.0865 THEN Rings=5 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.045 AND Shucked_weight>=0.0345 AND Viscera_weight>=0.041 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.029 AND Shucked_weight>=0.0345 AND Whole_weight>=0.2225 THEN Rings=8 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.029 AND Shucked_weight>=0.0345 AND Shucked_weight<=0.0425 AND Diameter>=0.21 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.029 AND Shucked_weight>=0.0345 AND Length<=0.275 AND Diameter>=0.215 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0295 AND Shucked_weight>=0.0345 AND Whole_weight>=0.2005 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Shucked_weight>=0.0345 AND Length<=0.275 AND Height>=0.07 THEN Rings=6 -0.0
    [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.032 AND Shucked_weight<=0.0345 AND Shell_weight>=0.021 AND Length>=0.27 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.048 AND Shucked_weight<=0.0255 AND Shucked_weight>=0.021 AND Diameter>=0.195 THEN Rings=6 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.048 AND Shucked_weight<=0.027 AND Diameter<=0.14 AND Diameter>=0.14 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.074 AND Shucked_weight<=0.027 AND Diameter>=0.165 AND Shucked_weight>=0.027 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.19 AND Diameter>=0.19 THEN Rings=6 -0.0
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0085 AND Diameter>=0.15 THEN Rings=3 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.195 AND Diameter>=0.195 AND Height>=0.065 THEN Rings=8 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.265 AND Length<=0.2 AND Diameter>=0.145 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Viscera_weight<=0.0205 AND Viscera_weight>=0.014 AND Diameter>=0.2 THEN Rings=7 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.024 AND Length<=0.205 AND Diameter>=0.15 THEN Rings=5 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.024 AND Length<=0.21 AND Diameter>=0.15 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.074 AND Length<=0.265 AND Diameter>=0.195 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.345 AND Shucked_weight>=0.0655 AND Height>=0.1 AND Diameter>=0.265 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Viscera_weight<=0.024 AND Diameter>=0.225 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.074 AND Viscera_weight<=0.0275 AND Length>=0.325 THEN Rings=6 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0275 AND Length<=0.215 AND Diameter>=0.155 THEN Rings=5 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0275 AND Length<=0.22 AND Diameter>=0.165 THEN Rings=5 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Shucked_weight>=0.0655 AND Shell_weight<=0.048 AND Height>=0.09 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.074 AND Viscera_weight<=0.0275 AND Viscera_weight>=0.0275 THEN Rings=7 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.015 AND Diameter>=0.16 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.048 AND Shucked_weight>=0.0675 AND Diameter>=0.265 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.048 AND Shucked_weight>=0.0675 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.049 AND Diameter>=0.24 AND Height>=0.08 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Shucked_weight>=0.0705 AND Viscera_weight>=0.0395 AND Diameter>=0.255 THEN Rings=7 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.255 AND Viscera_weight<=0.0295 AND Viscera_weight<=0.014 AND Length>=0.24 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.049 AND Shucked_weight>=0.0545 AND Shell_weight>=0.049 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.074 AND Whole_weight>=0.1655 AND Viscera_weight>=0.0505 AND Height>=0.1 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.075 AND Height>=0.095 AND Viscera_weight>=0.061 THEN Rings=11 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.05 AND Viscera_weight<=0.0285 AND Length>=0.255 AND Height>=0.1 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Diameter<=0.24 AND Whole_weight>=0.1715 AND Height>=0.085 THEN Rings=7 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Diameter<=0.245 AND Height<=0.06 AND Length<=0.255 AND Diameter>=0.18 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Diameter<=0.245 AND Height>=0.085 AND Viscera_weight>=0.0475 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1845 AND Shell_weight<=0.075 AND Height>=0.095 AND Whole_weight>=0.25 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Shell_weight<=0.059 AND Length<=0.28 AND Length>=0.28 AND Whole_weight>=0.127 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Shell_weight<=0.059 AND Length<=0.28 AND Diameter>=0.205 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Diameter<=0.25 AND Height>=0.085 AND Height>=0.105 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Diameter<=0.25 AND Height>=0.085 AND Shell_weight>=0.0795 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Length>=0.43 AND Diameter<=0.34 AND Shell_weight>=0.13 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Length>=0.43 AND Diameter<=0.34 AND Diameter>=0.34 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Length>=0.43 AND Shell_weight<=0.12 AND Whole_weight>=0.439 AND Height>=0.105 THEN Rings=7 -0.0
    [0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.1135 AND Shell_weight<=0.059 AND Viscera_weight>=0.034 AND Viscera_weight<=0.0375 AND Diameter>=0.215 THEN Rings=6 -0.7219280948873623
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Length>=0.5 AND Diameter>=0.39 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Shucked_weight>=0.197 AND Whole_weight>=0.525 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shell_weight<=0.059 AND Shucked_weight>=0.063 AND Shell_weight>=0.059 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shucked_weight>=0.1705 AND Shell_weight<=0.115 AND Height>=0.105 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Shell_weight>=0.123 AND Shell_weight>=0.1525 THEN Rings=9 -0.8112781244591328
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Viscera_weight>=0.086 AND Shucked_weight<=0.122 AND Diameter>=0.345 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Shucked_weight>=0.197 AND Diameter>=0.37 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Shucked_weight>=0.197 AND Diameter>=0.36 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Shucked_weight>=0.197 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Height>=0.105 AND Length>=0.435 AND Diameter>=0.365 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shell_weight>=0.06 AND Whole_weight>=0.488 AND Diameter>=0.44 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shell_weight>=0.06 AND Whole_weight>=0.488 AND Height>=0.1 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shell_weight>=0.06 AND Whole_weight>=0.488 AND Diameter>=0.365 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shucked_weight>=0.0705 AND Shucked_weight<=0.085 AND Viscera_weight>=0.0505 AND Diameter>=0.295 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shucked_weight>=0.0705 AND Shucked_weight<=0.085 AND Viscera_weight>=0.0505 AND Diameter>=0.27 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Viscera_weight>=0.0615 AND Length>=0.43 AND Viscera_weight>=0.125 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.105 AND Shell_weight>=0.06 AND Length<=0.37 AND Whole_weight>=0.251 AND Length>=0.36 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.295 AND Shell_weight>=0.06 AND Diameter<=0.25 AND Diameter>=0.25 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.295 AND Shell_weight>=0.06 AND Viscera_weight<=0.0355 AND Length>=0.35 THEN Rings=10 -0.0
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.38 AND Diameter<=0.3 AND Whole_weight<=0.144 AND Shell_weight<=0.035 AND Length>=0.3 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.3 AND Length<=0.31 AND Shucked_weight>=0.062 AND Diameter>=0.235 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.3 AND Length<=0.315 AND Length>=0.31 AND Diameter>=0.23 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.3 AND Viscera_weight<=0.036 AND Whole_weight>=0.1585 AND Diameter>=0.26 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.3 AND Shucked_weight<=0.0595 AND Diameter<=0.215 AND Diameter>=0.215 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.485 AND Diameter<=0.3 AND Shucked_weight<=0.0595 AND Shucked_weight>=0.0595 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.3 AND Shucked_weight<=0.06 AND Shucked_weight>=0.06 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.3 AND Length<=0.325 AND Shucked_weight<=0.0455 AND Diameter>=0.225 THEN Rings=9 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.3 AND Length<=0.325 AND Shell_weight<=0.037 AND Diameter>=0.22 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.3 AND Length<=0.325 AND Shucked_weight>=0.0825 THEN Rings=6 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.3 AND Length<=0.33 AND Shucked_weight>=0.065 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length<=0.33 AND Diameter>=0.255 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Height>=0.105 AND Shell_weight>=0.1265 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Height>=0.105 AND Shell_weight<=0.085 AND Diameter>=0.29 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Height>=0.105 AND Shell_weight<=0.0875 AND Whole_weight>=0.2905 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length<=0.335 AND Shucked_weight>=0.0735 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length<=0.335 AND Diameter>=0.25 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Viscera_weight<=0.037 AND Shucked_weight>=0.124 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Whole_weight>=0.2 AND Viscera_weight<=0.049 AND Diameter>=0.3 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Whole_weight>=0.2 AND Viscera_weight<=0.049 AND Whole_weight>=0.2295 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Whole_weight>=0.2 AND Length>=0.43 AND Shell_weight>=0.12 AND Length>=0.45 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Whole_weight>=0.2 AND Length>=0.43 AND Shell_weight>=0.125 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Whole_weight>=0.2 AND Length>=0.43 AND Diameter<=0.325 THEN Rings=7 -0.9182958340544896
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Viscera_weight<=0.037 AND Diameter>=0.26 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Viscera_weight<=0.038 AND Diameter>=0.28 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Height>=0.105 AND Shell_weight<=0.09 AND Diameter>=0.305 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Viscera_weight<=0.04 AND Length>=0.36 AND Whole_weight>=0.2255 THEN Rings=8 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Viscera_weight<=0.041 AND Length>=0.355 AND Diameter>=0.27 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0675 AND Shell_weight>=0.06 AND Whole_weight>=0.228 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0705 AND Shell_weight>=0.06 AND Diameter>=0.27 THEN Rings=9 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Shell_weight<=0.054 AND Length>=0.32 AND Length<=0.34 AND Diameter>=0.25 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Diameter<=0.25 AND Length>=0.345 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0705 AND Length>=0.355 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0705 AND Shucked_weight>=0.0705 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.077 AND Length>=0.325 AND Height>=0.09 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.077 AND Length<=0.295 AND Diameter>=0.225 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Length<=0.325 AND Length>=0.325 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Diameter<=0.25 AND Length>=0.34 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Shell_weight<=0.055 AND Length>=0.34 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0745 AND Length>=0.36 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.078 AND Length>=0.365 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.078 AND Length>=0.36 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.08 AND Diameter>=0.26 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0865 AND Diameter>=0.295 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0885 AND Height>=0.095 AND Diameter>=0.3 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0885 AND Height>=0.1 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0895 AND Height>=0.13 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.092 AND Height>=0.085 AND Shucked_weight>=0.092 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.093 AND Height>=0.095 AND Diameter>=0.305 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0975 AND Height>=0.11 AND Height>=0.14 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.0975 AND Height>=0.11 AND Shucked_weight>=0.0975 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1045 AND Height>=0.11 AND Shucked_weight>=0.1 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1045 AND Height>=0.11 AND Length>=0.37 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1045 AND Height>=0.085 AND Shucked_weight>=0.1045 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1065 AND Height>=0.085 AND Diameter>=0.31 THEN Rings=11 -0.0
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.112 AND Whole_weight<=0.1375 AND Length<=0.3 AND Diameter>=0.235 THEN Rings=4 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.112 AND Diameter<=0.255 AND Length<=0.32 AND Diameter>=0.245 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.112 AND Length<=0.35 AND Diameter>=0.26 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.112 AND Length<=0.35 AND Diameter>=0.255 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1375 AND Shucked_weight<=0.112 AND Shucked_weight<=0.09 AND Diameter>=0.275 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1375 AND Diameter<=0.335 AND Whole_weight>=0.4405 AND Height>=0.14 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Shell_weight<=0.1515 AND Length>=0.47 AND Whole_weight>=0.549 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Shell_weight<=0.1515 AND Viscera_weight>=0.1125 AND Shucked_weight>=0.222 AND Length>=0.465 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Length<=0.385 AND Diameter<=0.26 AND Diameter>=0.26 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Viscera_weight<=0.0525 AND Length<=0.36 AND Diameter>=0.275 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.345 AND Shucked_weight>=0.1965 AND Shucked_weight>=0.2355 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Viscera_weight<=0.0525 AND Viscera_weight>=0.0525 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Whole_weight<=0.2605 AND Shucked_weight>=0.112 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1375 AND Shucked_weight<=0.119 AND Viscera_weight>=0.0655 AND Whole_weight>=0.412 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.4425 AND Viscera_weight<=0.0545 AND Length>=0.37 AND Length>=0.375 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Shell_weight<=0.1515 AND Viscera_weight>=0.113 AND Length>=0.47 AND Diameter>=0.375 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Shell_weight<=0.1515 AND Viscera_weight>=0.113 AND Length>=0.47 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Viscera_weight<=0.0545 AND Diameter>=0.295 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Viscera_weight<=0.0565 AND Diameter>=0.325 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Viscera_weight<=0.0575 AND Viscera_weight<=0.0445 AND Diameter>=0.275 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Shell_weight<=0.096 AND Whole_weight>=0.3435 AND Height>=0.125 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.345 AND Shucked_weight>=0.1965 AND Whole_weight>=0.4885 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.345 AND Shucked_weight>=0.1785 AND Diameter>=0.345 AND Shucked_weight>=0.209 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.4425 AND Viscera_weight>=0.1005 AND Height>=0.125 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.345 AND Whole_weight>=0.443 AND Diameter>=0.345 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Diameter<=0.345 AND Whole_weight>=0.4165 AND Shell_weight>=0.185 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Diameter<=0.345 AND Whole_weight>=0.4165 AND Whole_weight>=0.4525 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.158 AND Shucked_weight>=0.225 AND Whole_weight<=0.48 AND Diameter>=0.36 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length>=0.415 AND Whole_weight<=0.3435 AND Diameter>=0.325 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2015 AND Viscera_weight>=0.1235 AND Shucked_weight<=0.1895 AND Shucked_weight>=0.1895 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length>=0.415 AND Height>=0.105 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length>=0.415 AND Whole_weight>=0.439 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2085 AND Shucked_weight>=0.2015 AND Height<=0.12 AND Diameter>=0.37 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2085 AND Diameter<=0.345 AND Length<=0.355 AND Whole_weight>=0.3275 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Length>=0.415 AND Height>=0.1 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2015 AND Shell_weight>=0.179 AND Diameter>=0.39 AND Diameter>=0.405 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2085 AND Shucked_weight>=0.2015 AND Diameter>=0.37 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2085 AND Shucked_weight>=0.2005 AND Height>=0.14 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shucked_weight<=0.21 AND Viscera_weight>=0.1235 AND Shucked_weight<=0.1725 AND Diameter>=0.365 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.234 AND Shell_weight>=0.18 AND Viscera_weight>=0.167 AND Whole_weight>=0.7535 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.29 AND Shucked_weight>=0.336 AND Diameter>=0.415 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.29 AND Whole_weight>=0.6785 AND Shell_weight>=0.195 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.29 AND Shucked_weight>=0.336 AND Shell_weight>=0.1885 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.21 AND Shell_weight<=0.145 AND Shucked_weight>=0.2025 AND Diameter>=0.35 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shell_weight<=0.1515 AND Whole_weight<=0.286 AND Diameter<=0.275 AND Diameter>=0.275 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2105 AND Diameter<=0.345 AND Shucked_weight>=0.1635 AND Shucked_weight>=0.1825 AND Length>=0.41 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.29 AND Shucked_weight>=0.336 AND Length>=0.51 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Length>=0.435 AND Shell_weight>=0.195 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.29 AND Diameter>=0.405 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.29 AND Whole_weight<=0.582 AND Whole_weight>=0.582 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Shucked_weight>=0.1705 AND Shucked_weight>=0.194 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Shucked_weight>=0.1705 AND Length>=0.415 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.2925 AND Shell_weight>=0.175 AND Shucked_weight>=0.336 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.2925 AND Diameter>=0.395 AND Height>=0.14 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight>=0.2925 AND Diameter>=0.395 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2035 AND Shell_weight<=0.0985 AND Viscera_weight>=0.0695 AND Whole_weight>=0.387 THEN Rings=11 -0.0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2035 AND Shell_weight<=0.1045 AND Shucked_weight>=0.1635 AND Height>=0.115 THEN Rings=5 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Viscera_weight>=0.085 AND Whole_weight>=0.38 AND Length>=0.435 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.305 AND Length>=0.41 AND Length>=0.415 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.32 AND Length<=0.36 AND Length<=0.35 AND Diameter>=0.275 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Viscera_weight>=0.085 AND Diameter>=0.32 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Length>=0.41 AND Height>=0.1 AND Shucked_weight>=0.1735 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Length<=0.36 AND Diameter>=0.285 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Shucked_weight>=0.127 AND Shell_weight>=0.12 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Shell_weight>=0.115 AND Viscera_weight>=0.091 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Viscera_weight<=0.051 AND Diameter>=0.265 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Shell_weight>=0.1 AND Shell_weight>=0.115 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.105 AND Shucked_weight>=0.1215 AND Shell_weight>=0.1 AND Diameter>=0.31 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.315 AND Shell_weight>=0.1105 AND Diameter>=0.3 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Height>=0.105 AND Diameter>=0.345 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Whole_weight>=0.304 AND Height>=0.11 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Height<=0.11 AND Height>=0.11 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.42 AND Whole_weight>=0.3675 AND Shell_weight>=0.148 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.325 AND Diameter>=0.325 AND Height>=0.12 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.325 AND Viscera_weight>=0.0695 AND Height<=0.09 AND Diameter>=0.31 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Diameter<=0.325 AND Viscera_weight>=0.07 AND Viscera_weight>=0.1065 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.24 AND Shucked_weight<=0.171 AND Shucked_weight<=0.103 AND Height>=0.095 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Diameter<=0.33 AND Length>=0.425 AND Length>=0.445 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.1565 AND Length>=0.44 AND Height>=0.14 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1565 AND Length>=0.44 AND Diameter>=0.355 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.33 AND Diameter>=0.33 AND Viscera_weight>=0.0935 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.33 AND Diameter>=0.31 AND Length>=0.435 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.33 AND Shell_weight>=0.1105 AND Diameter>=0.33 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.42 AND Whole_weight>=0.3675 AND Diameter>=0.34 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.42 AND Shell_weight>=0.1105 AND Diameter>=0.335 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.44 AND Shucked_weight>=0.175 AND Height>=0.135 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.44 AND Diameter>=0.35 AND Whole_weight>=0.516 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.44 AND Diameter>=0.35 AND Height>=0.125 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Viscera_weight<=0.0645 AND Viscera_weight>=0.0645 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Viscera_weight<=0.0655 AND Shucked_weight>=0.13 AND Height>=0.12 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.3615 AND Diameter>=0.31 AND Diameter>=0.35 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.415 AND Shucked_weight>=0.121 AND Diameter>=0.355 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Whole_weight<=0.421 AND Length>=0.45 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.425 AND Shucked_weight>=0.121 AND Whole_weight>=0.425 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.4425 AND Diameter>=0.34 AND Height>=0.13 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.4425 AND Diameter>=0.31 AND Length>=0.445 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.444 AND Shucked_weight>=0.121 AND Whole_weight>=0.444 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.447 AND Height>=0.115 AND Diameter>=0.36 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.4495 AND Height>=0.115 AND Diameter>=0.355 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.45 AND Height>=0.115 AND Diameter>=0.37 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.445 AND Diameter>=0.355 AND Diameter>=0.365 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.445 AND Diameter>=0.31 AND Length>=0.445 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1715 AND Diameter>=0.35 AND Diameter>=0.36 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.452 AND Diameter>=0.31 AND Length>=0.475 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.454 AND Diameter>=0.31 AND Length>=0.465 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.458 AND Shucked_weight>=0.121 AND Diameter>=0.355 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.46 AND Diameter>=0.31 AND Length>=0.45 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Shucked_weight<=0.1815 AND Whole_weight>=0.4835 AND Length>=0.52 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.466 AND Diameter>=0.34 AND Height>=0.13 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Length<=0.45 AND Whole_weight>=0.479 AND Shucked_weight>=0.2125 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.1845 AND Height>=0.115 AND Height>=0.145 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1815 AND Viscera_weight>=0.1375 AND Diameter>=0.36 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1815 AND Diameter>=0.34 AND Whole_weight>=0.4835 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1845 AND Height>=0.115 AND Shucked_weight>=0.121 AND Height>=0.125 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.1845 AND Height>=0.115 AND Length>=0.405 AND Whole_weight>=0.4705 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.185 AND Diameter>=0.34 AND Height>=0.125 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2 AND Shell_weight<=0.155 AND Length>=0.43 AND Diameter>=0.36 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.467 AND Shucked_weight>=0.1215 AND Diameter>=0.365 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Viscera_weight<=0.085 AND Shucked_weight>=0.1215 AND Diameter>=0.375 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.185 AND Diameter>=0.31 AND Length>=0.5 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.187 AND Diameter>=0.31 AND Length>=0.47 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.1925 AND Shucked_weight>=0.1215 AND Shell_weight<=0.095 AND Length>=0.41 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1915 AND Length>=0.405 AND Diameter>=0.37 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.1925 AND Whole_weight>=0.316 AND Diameter>=0.375 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.193 AND Length>=0.405 AND Diameter>=0.35 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.1975 AND Length>=0.405 AND Diameter>=0.39 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Shucked_weight<=0.198 AND Length>=0.405 AND Diameter>=0.36 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Diameter<=0.35 AND Diameter>=0.35 AND Whole_weight>=0.515 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.2 AND Diameter>=0.375 AND Diameter>=0.39 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.24 AND Length<=0.46 AND Diameter>=0.38 AND Height>=0.155 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Shucked_weight<=0.2 AND Length>=0.405 AND Diameter>=0.375 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.2035 AND Whole_weight>=0.5465 AND Height>=0.145 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2035 AND Shell_weight>=0.17 AND Length>=0.47 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2035 AND Whole_weight>=0.3185 AND Shucked_weight>=0.2035 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2105 AND Whole_weight>=0.3185 AND Shucked_weight>=0.2105 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.211 AND Whole_weight>=0.316 AND Diameter>=0.385 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Length<=0.47 AND Shucked_weight>=0.266 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2325 AND Diameter<=0.375 AND Length>=0.475 AND Height>=0.18 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Whole_weight<=0.5415 AND Diameter>=0.38 AND Diameter>=0.39 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Whole_weight<=0.5415 AND Diameter>=0.38 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Diameter<=0.375 AND Length>=0.475 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Shucked_weight<=0.2125 AND Whole_weight>=0.3185 AND Diameter>=0.4 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Shucked_weight<=0.2135 AND Whole_weight>=0.316 AND Diameter>=0.38 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Viscera_weight>=0.127 AND Height>=0.145 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Height>=0.13 AND Whole_weight>=0.673 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.233 AND Viscera_weight>=0.127 AND Diameter>=0.415 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.234 AND Shucked_weight<=0.216 AND Whole_weight>=0.3185 AND Height>=0.14 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.234 AND Viscera_weight>=0.1365 AND Height>=0.145 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.236 AND Viscera_weight>=0.1365 AND Height>=0.145 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.237 AND Viscera_weight>=0.1365 AND Diameter>=0.415 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2385 AND Whole_weight<=0.5415 AND Shell_weight>=0.135 AND Diameter>=0.395 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.5415 AND Length>=0.475 AND Diameter>=0.36 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.195 AND Whole_weight<=0.549 AND Length>=0.475 AND Diameter>=0.38 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Whole_weight<=0.5565 AND Height<=0.09 AND Length>=0.385 AND Height>=0.09 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2385 AND Shucked_weight>=0.234 AND Diameter>=0.44 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.239 AND Length>=0.475 AND Height<=0.12 AND Diameter>=0.4 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.18 AND Shucked_weight>=0.2925 AND Shucked_weight>=0.32 AND Length>=0.49 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2385 AND Height>=0.13 AND Shucked_weight>=0.235 AND Diameter>=0.385 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2385 AND Viscera_weight>=0.1365 AND Whole_weight>=0.669 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2385 AND Viscera_weight>=0.1365 AND Height>=0.145 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.239 AND Height>=0.13 AND Viscera_weight>=0.1365 AND Shucked_weight>=0.231 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.239 AND Length>=0.475 AND Diameter<=0.38 AND Length>=0.49 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.239 AND Length>=0.51 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.239 AND Length>=0.475 AND Diameter>=0.395 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.24 AND Diameter>=0.39 AND Diameter>=0.41 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2405 AND Length>=0.525 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2405 AND Diameter>=0.38 AND Height>=0.155 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Shucked_weight<=0.2415 AND Height>=0.14 AND Viscera_weight>=0.14 THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2415 AND Height>=0.13 AND Whole_weight>=0.6155 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shucked_weight<=0.2415 AND Whole_weight>=0.5565 AND Height>=0.15 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2415 AND Diameter>=0.375 AND Length>=0.47 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.242 AND Height>=0.14 AND Diameter>=0.4 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.242 AND Length>=0.475 AND Diameter>=0.375 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2465 AND Diameter>=0.39 AND Length>=0.51 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2465 AND Viscera_weight>=0.126 AND Diameter>=0.4 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.245 AND Length>=0.475 AND Height>=0.13 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.246 AND Length>=0.475 AND Diameter>=0.385 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2465 AND Length>=0.475 AND Diameter>=0.39 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Shucked_weight<=0.249 AND Whole_weight>=0.66 AND Length>=0.545 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2475 AND Diameter>=0.39 AND Diameter>=0.395 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.249 AND Diameter>=0.38 AND Length>=0.505 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.175 AND Length>=0.51 AND Height>=0.125 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.175 AND Whole_weight>=0.6515 AND Length>=0.52 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.249 AND Shucked_weight>=0.23 AND Diameter>=0.415 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Shucked_weight<=0.2505 AND Height>=0.145 AND Height>=0.165 THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.175 AND Whole_weight<=0.283 AND Length>=0.39 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.249 AND Length>=0.475 AND Diameter>=0.38 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2505 AND Length>=0.55 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.16 AND Shell_weight>=0.1565 AND Whole_weight>=0.64 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.252 AND Length>=0.475 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.252 AND Whole_weight<=0.2535 AND Diameter>=0.295 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Diameter>=0.4 AND Whole_weight>=0.943 AND Diameter>=0.455 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.136 AND Diameter>=0.4 AND Diameter>=0.41 AND Whole_weight>=0.9815 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Whole_weight<=0.2895 AND Length<=0.38 AND Diameter>=0.29 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.155 AND Shell_weight>=0.135 AND Diameter>=0.41 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Shell_weight<=0.095 AND Length>=0.385 AND Diameter>=0.31 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Viscera_weight<=0.063 AND Length<=0.385 AND Diameter>=0.3 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Viscera_weight<=0.0635 AND Length<=0.385 AND Diameter>=0.29 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Viscera_weight<=0.07 AND Length<=0.395 AND Diameter>=0.295 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Viscera_weight<=0.07 AND Diameter>=0.3 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Length<=0.395 AND Length>=0.395 AND Shucked_weight>=0.138 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Whole_weight<=0.304 AND Length<=0.395 AND Diameter>=0.29 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.252 AND Shucked_weight>=0.23 AND Whole_weight>=0.5615 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.2535 AND Shucked_weight>=0.23 AND Height>=0.15 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.145 AND Height>=0.15 AND Whole_weight>=0.8365 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.2555 AND Shucked_weight>=0.2555 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.256 AND Shucked_weight>=0.256 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shucked_weight<=0.2565 AND Diameter>=0.4 AND Length>=0.53 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Shell_weight>=0.195 AND Shucked_weight>=0.302 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.727 AND Height>=0.145 AND Shell_weight>=0.25 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.145 AND Height>=0.145 AND Diameter>=0.45 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.137 AND Height>=0.145 AND Shell_weight>=0.225 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.145 AND Height>=0.145 AND Length>=0.53 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.145 AND Height>=0.145 AND Diameter>=0.41 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.146 AND Length>=0.515 AND Diameter<=0.4 AND Diameter>=0.4 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.6995 AND Height>=0.145 AND Height>=0.155 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.146 AND Height<=0.095 AND Length<=0.4 AND Diameter>=0.305 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.146 AND Height<=0.115 AND Length>=0.48 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.16 AND Shell_weight>=0.1405 AND Height>=0.15 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.17 AND Length<=0.405 AND Length>=0.405 AND Whole_weight>=0.3485 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.16 AND Shell_weight>=0.1405 AND Length>=0.465 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.083 AND Diameter>=0.345 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.0905 AND Diameter>=0.305 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.254 AND Shucked_weight>=0.254 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.255 AND Shucked_weight>=0.255 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2565 AND Shucked_weight>=0.2565 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1415 AND Viscera_weight<=0.095 AND Diameter>=0.34 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1415 AND Whole_weight<=0.517 AND Shucked_weight>=0.23 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1415 AND Viscera_weight<=0.114 AND Length>=0.48 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1415 AND Length>=0.485 AND Viscera_weight>=0.1415 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1425 AND Whole_weight>=0.7635 AND Diameter>=0.405 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.275 AND Viscera_weight<=0.1315 AND Shell_weight>=0.22 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.272 AND Height>=0.135 AND Diameter>=0.425 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.272 AND Height>=0.14 AND Height>=0.16 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1425 AND Whole_weight>=0.6645 AND Viscera_weight>=0.1425 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.146 AND Whole_weight>=0.6645 AND Shucked_weight>=0.3165 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.275 AND Height>=0.14 AND Diameter>=0.425 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.276 AND Whole_weight<=0.659 AND Viscera_weight>=0.179 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2725 AND Shell_weight>=0.205 AND Diameter>=0.425 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2725 AND Shell_weight>=0.205 AND Whole_weight>=0.6995 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2725 AND Shell_weight>=0.205 AND Diameter>=0.405 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.276 AND Whole_weight>=0.927 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.277 AND Whole_weight>=0.8425 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.277 AND Whole_weight>=0.77 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.277 AND Shell_weight>=0.205 AND Diameter>=0.42 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.277 AND Shell_weight>=0.205 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.146 AND Shell_weight>=0.21 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1495 AND Shell_weight>=0.31 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1495 AND Shell_weight>=0.26 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.277 AND Viscera_weight>=0.126 AND Viscera_weight>=0.133 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.673 AND Shucked_weight>=0.784 AND Diameter>=0.58 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.325 AND Whole_weight<=0.612 AND Diameter>=0.37 AND Length>=0.485 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.165 AND Viscera_weight<=0.15 AND Diameter>=0.375 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3135 AND Diameter<=0.4 AND Whole_weight>=0.823 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3135 AND Diameter<=0.405 AND Shucked_weight>=0.3075 AND Diameter>=0.405 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3135 AND Length>=0.535 AND Shell_weight>=0.335 AND Height>=0.195 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.325 AND Length<=0.49 AND Diameter>=0.38 AND Diameter>=0.39 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.313 AND Diameter<=0.4 AND Diameter>=0.38 AND Whole_weight>=0.775 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Length<=0.495 AND Diameter>=0.38 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.325 AND Whole_weight<=0.612 AND Shucked_weight>=0.327 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Shucked_weight>=0.784 AND Length>=0.705 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Shucked_weight>=0.784 AND Diameter>=0.56 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Length<=0.5 AND Shucked_weight>=0.346 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Diameter<=0.38 AND Shucked_weight>=0.325 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Length<=0.505 AND Shucked_weight>=0.3415 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Viscera_weight>=0.3695 AND Viscera_weight>=0.541 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Shucked_weight>=0.7665 AND Shell_weight>=0.78 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Length<=0.675 AND Diameter>=0.545 AND Diameter>=0.565 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Length<=0.675 AND Height>=0.23 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Height>=0.21 AND Diameter>=0.59 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Shucked_weight>=0.784 AND Diameter>=0.55 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Height>=0.21 AND Whole_weight>=2.141 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Height>=0.21 AND Length>=0.725 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.719 AND Height>=0.21 AND Length>=0.68 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]  IF Whole_weight>=1.7255 AND Height>=0.215 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.152 AND Shell_weight>=0.32 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.152 AND Shucked_weight>=0.3515 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1525 AND Viscera_weight<=0.11 AND Diameter>=0.36 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Length<=0.51 AND Diameter>=0.41 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.625 AND Diameter>=0.545 AND Shell_weight>=0.58 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2825 AND Shell_weight>=0.335 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2825 AND Whole_weight>=0.8665 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.1525 AND Shell_weight<=0.14 AND Diameter>=0.365 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2855 AND Length<=0.455 AND Diameter>=0.355 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2855 AND Length<=0.46 AND Diameter>=0.355 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2855 AND Length<=0.485 AND Diameter>=0.365 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2865 AND Length<=0.51 AND Diameter>=0.4 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.291 AND Length>=0.525 AND Height>=0.15 AND Height>=0.17 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.292 AND Length<=0.515 AND Diameter>=0.435 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Length<=0.47 AND Diameter>=0.37 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Length<=0.525 AND Whole_weight>=0.8175 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Length<=0.52 AND Diameter>=0.41 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.295 AND Diameter>=0.445 AND Diameter>=0.455 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.295 AND Shell_weight<=0.214 AND Diameter>=0.41 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.295 AND Shucked_weight>=0.289 AND Length>=0.545 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Shucked_weight<=0.2785 AND Diameter>=0.42 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.2935 AND Viscera_weight>=0.1595 AND Length>=0.545 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Viscera_weight>=0.225 AND Length>=0.575 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3325 AND Shucked_weight>=0.325 AND Height>=0.235 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.335 AND Shucked_weight>=0.325 AND Shell_weight<=0.21 AND Diameter>=0.425 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Shucked_weight<=0.335 AND Length>=0.555 AND Length>=0.585 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.335 AND Shucked_weight<=0.293 AND Whole_weight>=0.733 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.335 AND Shucked_weight<=0.2935 AND Length>=0.535 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Height>=0.195 AND Diameter>=0.505 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Viscera_weight>=0.2565 AND Shucked_weight>=0.586 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.325 AND Shell_weight<=0.23 AND Diameter>=0.44 AND Diameter>=0.46 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.325 AND Shell_weight<=0.23 AND Diameter>=0.44 AND Whole_weight>=0.93 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1375 AND Shucked_weight>=0.4815 AND Height>=0.16 AND Length>=0.6 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1375 AND Shucked_weight>=0.4815 AND Height>=0.16 AND Diameter>=0.46 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Diameter>=0.49 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Whole_weight<=1.012 AND Diameter>=0.455 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Whole_weight<=1.0635 AND Diameter>=0.445 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Whole_weight<=1.065 AND Length>=0.54 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  IF Shucked_weight>=0.628 AND Height>=0.185 AND Whole_weight>=1.8075 AND Diameter>=0.585 THEN Rings=29 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.335 AND Shell_weight<=0.235 AND Shucked_weight>=0.3345 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.335 AND Shell_weight<=0.235 AND Shucked_weight>=0.32 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.335 AND Shucked_weight>=0.3275 AND Shucked_weight>=0.335 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Shell_weight<=0.235 AND Whole_weight>=0.7635 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Shell_weight<=0.24 AND Length>=0.535 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.926 AND Length>=0.575 AND Whole_weight>=0.922 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.171 AND Diameter>=0.44 AND Shucked_weight>=0.4145 AND Whole_weight>=1.21 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.171 AND Diameter>=0.44 AND Shucked_weight>=0.4145 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.926 AND Length>=0.575 AND Shucked_weight>=0.383 AND Diameter>=0.485 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.926 AND Length>=0.575 AND Diameter>=0.455 AND Length>=0.58 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.171 AND Length>=0.555 AND Viscera_weight>=0.1665 AND Diameter>=0.45 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.342 AND Shell_weight<=0.2475 AND Length<=0.525 AND Diameter>=0.41 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Whole_weight<=0.766 AND Length>=0.555 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Height<=0.145 AND Whole_weight>=0.843 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.339 AND Height<=0.145 AND Length>=0.55 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Shucked_weight>=0.3275 AND Whole_weight>=1.042 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Shucked_weight>=0.3275 AND Diameter>=0.44 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.155 AND Shucked_weight>=0.54 AND Viscera_weight>=0.2565 AND Whole_weight>=1.4785 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.155 AND Shucked_weight<=0.33 AND Length>=0.55 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.336 AND Length>=0.54 AND Diameter>=0.465 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.33 AND Length>=0.54 AND Diameter>=0.455 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Height>=0.185 AND Height>=0.205 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3175 AND Height<=0.16 AND Height>=0.16 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.314 AND Height>=0.185 AND Diameter>=0.42 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.44 AND Shucked_weight<=0.314 AND Shucked_weight>=0.314 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.339 AND Diameter>=0.45 AND Diameter>=0.465 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.341 AND Diameter>=0.445 AND Height>=0.165 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.341 AND Whole_weight>=0.97 AND Diameter>=0.45 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.341 AND Whole_weight>=0.97 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.342 AND Whole_weight>=0.9065 AND Whole_weight>=1.013 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3465 AND Whole_weight>=0.9065 AND Whole_weight>=0.909 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.342 AND Shell_weight<=0.2425 AND Diameter>=0.455 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.342 AND Shucked_weight>=0.33 AND Whole_weight>=0.9065 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3465 AND Shucked_weight>=0.33 AND Diameter>=0.445 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3175 AND Shucked_weight>=0.3175 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.341 AND Diameter>=0.425 AND Shucked_weight>=0.341 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3475 AND Diameter>=0.43 AND Shucked_weight>=0.3475 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.348 AND Diameter>=0.43 AND Shucked_weight>=0.348 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.354 AND Shell_weight<=0.235 AND Diameter>=0.42 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.354 AND Shell_weight<=0.25 AND Whole_weight>=0.8375 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.354 AND Whole_weight>=0.915 AND Height>=0.185 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.354 AND Viscera_weight>=0.2185 AND Shell_weight>=0.3 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.366 AND Whole_weight>=0.9325 AND Whole_weight>=1.0325 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.366 AND Whole_weight>=0.915 AND Diameter>=0.46 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.366 AND Whole_weight>=0.915 AND Whole_weight>=0.9555 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.367 AND Whole_weight>=1.1375 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.367 AND Whole_weight>=0.915 AND Diameter>=0.5 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.374 AND Whole_weight>=0.915 AND Diameter<=0.44 AND Diameter>=0.44 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Viscera_weight>=0.2185 AND Length>=0.59 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3665 AND Viscera_weight>=0.204 AND Diameter>=0.435 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3665 AND Whole_weight>=0.895 AND Whole_weight>=0.9325 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Shucked_weight>=0.3665 AND Diameter>=0.455 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Whole_weight>=0.892 AND Viscera_weight>=0.186 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Length>=0.565 AND Diameter>=0.455 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Length>=0.565 AND Whole_weight>=0.9 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Diameter>=0.44 AND Length>=0.565 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3685 AND Shell_weight<=0.205 AND Diameter>=0.41 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3685 AND Diameter>=0.435 AND Diameter>=0.45 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3685 AND Shucked_weight>=0.3465 AND Diameter>=0.44 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shell_weight<=0.235 AND Length<=0.515 AND Diameter>=0.43 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.473 AND Whole_weight>=1.217 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.369 AND Shucked_weight>=0.3465 AND Shucked_weight>=0.369 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight<=0.3695 AND Shucked_weight>=0.3465 AND Diameter>=0.43 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Diameter>=0.465 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Diameter>=0.46 THEN Rings=7 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Shucked_weight>=0.581 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Shucked_weight>=0.479 AND Whole_weight>=1.1935 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shell_weight<=0.235 AND Shell_weight>=0.235 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shell_weight<=0.2375 AND Whole_weight>=0.9185 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Viscera_weight<=0.1815 AND Length>=0.575 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shucked_weight<=0.3695 AND Diameter>=0.435 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shucked_weight<=0.3695 AND Viscera_weight>=0.2135 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9365 AND Diameter>=0.44 AND Height>=0.16 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shucked_weight<=0.3695 AND Height>=0.18 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.44 AND Whole_weight>=0.9885 AND Shucked_weight>=0.486 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.224 AND Shucked_weight<=0.378 AND Diameter>=0.425 AND Diameter>=0.47 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Shucked_weight<=0.378 AND Diameter>=0.435 AND Diameter>=0.45 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Viscera_weight>=0.2065 AND Height>=0.165 AND Diameter>=0.435 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Diameter>=0.44 AND Length>=0.61 AND Diameter>=0.49 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Viscera_weight<=0.188 AND Shucked_weight>=0.4035 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Height<=0.135 AND Shell_weight>=0.27 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Viscera_weight<=0.192 AND Diameter<=0.415 AND Diameter>=0.415 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.224 AND Viscera_weight<=0.1765 AND Height>=0.16 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2055 AND Diameter<=0.415 AND Diameter>=0.415 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Diameter>=0.44 AND Length>=0.58 AND Whole_weight>=0.97 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9395 AND Whole_weight>=0.883 AND Shucked_weight>=0.4275 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.941 AND Diameter>=0.44 AND Whole_weight>=0.941 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.952 AND Diameter>=0.44 AND Whole_weight>=0.952 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9535 AND Whole_weight>=0.883 AND Shucked_weight>=0.4465 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.967 AND Whole_weight>=0.883 AND Height>=0.18 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.967 AND Viscera_weight<=0.1695 AND Diameter>=0.42 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.967 AND Shucked_weight>=0.381 AND Height>=0.16 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9715 AND Diameter>=0.445 AND Whole_weight>=0.9715 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Viscera_weight<=0.188 AND Length>=0.57 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Diameter>=0.445 AND Whole_weight>=0.9665 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Diameter>=0.44 AND Diameter>=0.445 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=0.9885 AND Whole_weight>=0.926 AND Height>=0.165 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.029 AND Shell_weight<=0.245 AND Length>=0.575 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.455 AND Height>=0.175 AND Whole_weight>=1.3305 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.455 AND Shucked_weight>=0.5135 AND Height>=0.155 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.029 AND Shell_weight<=0.245 AND Whole_weight>=0.8215 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.029 AND Length>=0.585 AND Shucked_weight>=0.4425 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.029 AND Height>=0.175 AND Diameter>=0.475 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.029 AND Length>=0.585 AND Whole_weight>=1.029 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]  IF Shucked_weight<=0.392 AND Height>=0.195 AND Viscera_weight>=0.27 THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Height>=0.17 AND Viscera_weight>=0.261 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Height>=0.175 AND Shell_weight>=0.345 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Whole_weight>=1.092 AND Shucked_weight>=0.54 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Whole_weight>=1.092 AND Diameter>=0.46 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Length>=0.59 AND Viscera_weight>=0.3015 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Diameter>=0.46 AND Height>=0.19 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Whole_weight<=1.036 AND Length>=0.585 AND Whole_weight>=1.036 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Length>=0.59 AND Whole_weight>=1.098 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Viscera_weight>=0.2355 AND Whole_weight>=1.0235 AND Diameter>=0.46 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Viscera_weight>=0.2355 AND Whole_weight>=1.0235 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Whole_weight>=1.0575 AND Diameter>=0.465 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Whole_weight>=1.0575 AND Shell_weight>=0.325 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Whole_weight>=1.0575 AND Whole_weight>=1.068 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Diameter<=0.455 AND Length>=0.58 AND Height>=0.155 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Shucked_weight>=0.429 AND Whole_weight>=1.0595 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.455 AND Viscera_weight>=0.228 AND Diameter>=0.455 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Viscera_weight>=0.2385 AND Diameter>=0.46 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Shell_weight>=0.3 AND Diameter>=0.45 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Height>=0.17 AND Shucked_weight>=0.3895 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Shell_weight>=0.295 AND Shucked_weight>=0.3865 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Shell_weight>=0.3 AND Height>=0.18 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.46 AND Shell_weight>=0.3 AND Whole_weight>=0.9535 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Shell_weight>=0.275 AND Length>=0.57 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.58 AND Length<=0.515 AND Diameter>=0.425 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.585 AND Length>=0.58 AND Diameter>=0.465 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.053 AND Length>=0.58 AND Diameter>=0.525 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.59 AND Whole_weight>=1.053 AND Height>=0.2 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Whole_weight>=1.7255 AND Viscera_weight>=0.392 AND Length>=0.685 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.58 AND Whole_weight<=0.791 AND Diameter>=0.42 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Shucked_weight>=0.675 AND Whole_weight>=1.8095 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.195 AND Whole_weight>=1.787 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.195 AND Diameter>=0.555 AND Height>=0.2 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.053 AND Length<=0.535 AND Diameter>=0.42 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.195 AND Diameter>=0.555 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.195 AND Shucked_weight>=0.673 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.195 AND Shell_weight>=0.49 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.133 AND Shell_weight>=0.36 AND Length>=0.605 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.415 AND Viscera_weight>=0.358 AND Whole_weight>=1.6675 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Shucked_weight>=0.69 AND Viscera_weight>=0.3885 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Shucked_weight>=0.7595 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Height>=0.215 AND Length>=0.645 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.415 AND Shell_weight>=0.53 AND Diameter>=0.535 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.053 AND Length>=0.58 AND Diameter>=0.485 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Shucked_weight>=0.755 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Diameter>=0.535 AND Length>=0.68 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.415 AND Whole_weight>=1.5675 AND Diameter>=0.535 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Height<=0.17 AND Height>=0.17 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2435 AND Height>=0.19 AND Whole_weight>=1.219 AND Diameter>=0.535 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.2435 AND Shucked_weight>=0.5225 AND Length>=0.625 AND Diameter>=0.515 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.41 AND Shucked_weight>=0.6645 AND Diameter>=0.545 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.4225 AND Whole_weight>=1.626 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.4225 AND Shucked_weight>=0.69 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.4225 AND Whole_weight>=1.5675 AND Length>=0.645 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.4225 AND Shell_weight>=0.47 AND Viscera_weight>=0.358 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.4225 AND Whole_weight>=1.5165 AND Length>=0.63 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Height>=0.195 AND Whole_weight>=1.484 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  IF Height>=0.195 AND Whole_weight>=1.4455 THEN Rings=22 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height<=0.155 AND Shell_weight>=0.35 AND Diameter>=0.515 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.415 AND Whole_weight>=1.4225 AND Height>=0.18 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.5225 AND Shell_weight>=0.42 AND Diameter>=0.535 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.415 AND Length>=0.655 AND Length>=0.7 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.415 AND Length>=0.655 AND Diameter>=0.54 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Shucked_weight>=0.5295 AND Diameter>=0.53 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.5 AND Length>=0.66 AND Shucked_weight>=0.5515 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.5 AND Length>=0.66 AND Diameter>=0.585 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Height>=0.19 AND Length>=0.65 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Height>=0.19 AND Height>=0.205 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Height>=0.19 AND Diameter>=0.515 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Viscera_weight>=0.272 AND Diameter>=0.515 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.465 AND Length>=0.59 AND Height>=0.19 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.465 AND Length>=0.58 AND Shucked_weight>=0.5175 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Shell_weight>=0.48 AND Viscera_weight>=0.3035 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Viscera_weight>=0.266 AND Diameter>=0.51 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.2715 AND Length>=0.635 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.285 AND Height>=0.19 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.5 AND Length>=0.66 AND Viscera_weight>=0.298 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.285 AND Height>=0.16 AND Height>=0.185 THEN Rings=8 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.285 AND Height>=0.16 AND Diameter>=0.485 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.285 AND Shucked_weight>=0.429 AND Height>=0.16 THEN Rings=6 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Viscera_weight>=0.2635 AND Diameter>=0.5 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.465 AND Length>=0.595 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.465 AND Length>=0.58 AND Diameter>=0.465 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Diameter>=0.495 AND Viscera_weight>=0.2625 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  IF Whole_weight<=1.077 AND Height>=0.165 AND Diameter>=0.495 THEN Rings=26 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Length>=0.62 AND Length>=0.64 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.285 AND Length<=0.55 AND Diameter>=0.425 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.47 AND Viscera_weight<=0.188 AND Diameter>=0.44 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.47 AND Viscera_weight<=0.1975 AND Diameter>=0.425 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.513 AND Height>=0.185 AND Height>=0.2 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Diameter>=0.5 AND Shucked_weight>=0.5875 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.513 AND Height>=0.185 AND Height>=0.19 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.513 AND Shucked_weight>=0.586 AND Height>=0.185 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight>=0.2875 AND Diameter>=0.51 AND Diameter>=0.55 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight>=0.2875 AND Diameter>=0.51 AND Diameter>=0.525 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight>=0.291 AND Diameter>=0.51 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.175 AND Diameter>=0.54 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Diameter>=0.52 AND Viscera_weight>=0.2635 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Diameter>=0.52 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Diameter>=0.515 AND Height>=0.17 THEN Rings=16 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Whole_weight>=1.376 AND Length>=0.63 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  IF Diameter<=0.47 AND Shell_weight<=0.26 AND Diameter>=0.46 THEN Rings=19 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Length>=0.625 AND Whole_weight>=1.3135 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Diameter>=0.5 AND Whole_weight>=1.377 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight>=0.42 AND Length>=0.625 AND Height>=0.19 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.2895 AND Whole_weight>=1.382 AND Height>=0.19 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  IF Whole_weight>=1.2895 AND Shell_weight>=0.46 AND Diameter>=0.485 THEN Rings=23 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Length<=0.6 AND Height>=0.175 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Height>=0.19 AND Diameter>=0.505 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Height>=0.19 AND Height>=0.205 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Height>=0.18 AND Height>=0.19 AND Diameter>=0.485 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Height>=0.19 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  IF Shell_weight>=0.425 AND Height<=0.17 AND Whole_weight>=1.3175 THEN Rings=21 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shell_weight<=0.285 AND Shucked_weight>=0.443 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Viscera_weight<=0.2355 AND Diameter>=0.48 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.077 AND Shucked_weight>=0.4545 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1015 AND Length>=0.63 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1015 AND Whole_weight>=1.1015 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1085 AND Shucked_weight>=0.4695 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1185 AND Shucked_weight>=0.469 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1235 AND Shucked_weight>=0.428 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.1085 AND Height>=0.185 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Height>=0.185 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Whole_weight>=1.227 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Diameter>=0.49 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Height>=0.18 AND Length>=0.615 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.133 AND Height>=0.18 THEN Rings=13 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.133 AND Height>=0.165 THEN Rings=18 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight<=1.133 AND Height>=0.15 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2325 AND Length>=0.65 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2355 AND Length>=0.61 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2405 AND Height>=0.175 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Whole_weight>=1.262 AND Shucked_weight>=0.6355 THEN Rings=9 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  IF Viscera_weight<=0.2405 AND Length<=0.595 AND Diameter>=0.475 THEN Rings=20 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Viscera_weight>=0.297 AND Height>=0.175 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.615 AND Shucked_weight>=0.5265 AND Length>=0.62 THEN Rings=11 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Height>=0.17 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Viscera_weight>=0.301 THEN Rings=10 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Shucked_weight>=0.4975 AND Length>=0.605 THEN Rings=12 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.61 THEN Rings=14 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.6 THEN Rings=15 -0.0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.595 AND Whole_weight>=1.262 THEN Rings=17 -0.0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length>=0.595 THEN Rings=8 -0.0
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  IF Length<=0.075 THEN Rings=1 -0.0
    [1, 1, 7, 18, 31, 60, 97, 84, 109, 117, 91, 79, 71, 49, 52, 29, 28, 18, 17, 16, 9, 6, 5, 1, 1]  IF TRUE THEN Rings=10 -3.984037499126841

""" |> orange_decision_list

bs = BeamSearch( beam_width = 5, discretizedomain=true)
println(bs)

train_slice = 1:1000
test_slice = 1001:1500

table = CSV.read("datasets/yeast.csv", DataFrame)

############################################################################################
########################## Test with categorical attributes ################################
############################################################################################

y = table[:, :localization_site] |> CategoricalArray
X = select(table, Not([:localization_site, :Sequence_Name]));

# Train
X_train, y_train = preprocess_inputdata(X[train_slice, :],y[train_slice])
Xpl_train = PropositionalLogiset(X_train)

println("sequentialcovering ...")
abalone1000_dl_sole = sequentialcovering(Xpl_train, y_train, searchmethod = bs)

# Test
X_test, y_test = preprocess_inputdata(X[test_slice, :],y[test_slice])
Xpl_test = PropositionalLogiset(X_test)

yhat_sole   = apply(abalone1000_dl_sole, Xpl_test)
printstyled("Partial training (0.7) accuracy: ",
        trunc(MLJ.accuracy(y_test, yhat_sole), digits=3),"\n",
        color=:blue,bold=true)

yhat_orange = tryparse.(Int, apply(abalone1000_dl_orange, Xpl_test))
printstyled("Partial training (0.7) accuracy: ",
        trunc(MLJ.accuracy(y_test, yhat_orange), digits=3),"\n",
        color=:light_yellow,bold=true)


############################################################################################
########################## Test with no categorical attributes #############################
############################################################################################

y = table[:, :localization_site] |> CategoricalArray
X = select(table, Not([:localization_site, :Sex]));

# Train
X_train, y_train = preprocess_inputdata(X[train_slice, :],y[train_slice])
Xpl_train = PropositionalLogiset(X_train)

println("sequentialcovering (no categorical) ...")
abalone1000_no_categorical_dl_sole = sequentialcovering(Xpl_train, y_train, searchmethod = bs)

# Test
X_test, y_test = preprocess_inputdata(X[test_slice, :],y[test_slice])
Xpl_test = PropositionalLogiset(X_test)

yhat_sole = apply(abalone1000_no_categorical_dl_sole, Xpl_test)
printstyled("Partial training (0.7) accuracy: ",
        trunc(MLJ.accuracy(y_test, yhat_sole), digits=3),"\n",
        color=:blue,bold=true)

yhat_orange = tryparse.(Int, apply(abalone1000_no_categorical_dl_orange, Xpl_test))
printstyled("Partial training (0.7) accuracy: ",
        trunc(MLJ.accuracy(y_test, yhat_orange), digits=8),"\n",
        color=:light_yellow,bold=true)
