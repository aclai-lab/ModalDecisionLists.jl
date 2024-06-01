using MLJ: load_iris
using Test
using Random
using Test
using Random
using DataFrames
using CategoricalArrays
using RDatasets

using SoleBase: CLabel
using ModalDecisionLists
using ModalDecisionLists: maptointeger
using ModalDecisionLists.LossFunctions: entropy, laplace_accuracy
using MLJ

X...,y = MLJ.load_iris()
X = PropositionalLogiset(DataFrame(X))
y = Vector{CLabel}(y)

# TODO strategia pesata

# === Cardinality ==========================================================================

rs = RandSearch(cardinality = -1)
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 0)
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)
#
rs = RandSearch(cardinality = 1, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 50, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 100, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

# Parity case
rs = RandSearch(cardinality = 1, rng = MersenneTwister(13))
@test_logs (:warn,"Parity encountered in bestguess! counts (150 elements):" *
            " Dict(2 => 50, 3 => 50, 1 => 50), argmax: 2, max: 50 (sum = 150)"
    ) sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(cardinality = 2, rng = MersenneTwister(1))
@test_logs (:warn,"Parity encountered in bestguess! counts (142 elements):" *
            " Dict(2 => 48, 3 => 46, 1 => 48), argmax: 2, max: 48 (sum = 142)"
    ) sequentialcovering(X, y, searchmethod=rs)

# === Cardinality - Laplace accurcy ========================================================

rs = RandSearch(cardinality = 1, rng = MersenneTwister(78), loss_function=laplace_accuracy)
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(78), loss_function=laplace_accuracy)
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(79), loss_function=laplace_accuracy)
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 100, rng = MersenneTwister(78), loss_function=laplace_accuracy)
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

# Parity case
rs = RandSearch(cardinality = 50, rng = MersenneTwister(78), loss_function=laplace_accuracy)
@test_logs (:warn,"Parity encountered in bestguess! counts (8 elements):" *
            " Dict(2 => 4, 1 => 4), argmax: 2, max: 4 (sum = 8)"
    ) sequentialcovering(X, y, searchmethod=rs)

# === Operators ============================================================================

rs = RandSearch(cardinality = 5, rng = MersenneTwister(78),
                operators=[NEGATION])
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(78),
                operators=[NEGATION, DISJUNCTION])
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(78),
                operators=[NEGATION, CONJUNCTION])
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(78),
                operators=[NEGATION, CONJUNCTION, DISJUNCTION])
@test_nowarn sequentialcovering(X, y, searchmethod=rs)
##
rs = RandSearch(cardinality = 1, rng = MersenneTwister(78),
                operators=[])
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)
rs = RandSearch(cardinality = 5, rng = MersenneTwister(78),
                operators=[NEGATION, CONJUNCTION, "NOT_AN_OPERATOR"])
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)

# === Syntaxheight =========================================================================

rs = RandSearch(syntaxheight = -1, rng = MersenneTwister(78))
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(syntaxheight = 0, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(syntaxheight = 2, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(syntaxheight = 5, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)


# === Max purity =========================================================================

rs = RandSearch(max_purity_const = -0.5, rng = MersenneTwister(78))
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(max_purity_const = 2.0, rng = MersenneTwister(78))
@test_throws AssertionError sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(max_purity_const = 0.001, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(max_purity_const = 0.05, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(max_purity_const = 0.1, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(max_purity_const = 0.5, rng = MersenneTwister(78))
@test_nowarn sequentialcovering(X, y, searchmethod=rs)

rs = RandSearch(max_purity_const = 0.8, rng = MersenneTwister(78))
@test_logs (:warn,"Parity encountered in bestguess! counts (116 elements):" *
            " Dict(2 => 50, 3 => 16, 1 => 50), argmax: 2, max: 50 (sum = 116)")
rs = RandSearch(max_purity_const = 1.0, rng = MersenneTwister(78))
@test_logs (:warn,"Parity encountered in bestguess! counts (150 elements):" *
            " Dict(2 => 50, 3 => 50, 1 => 50), argmax: 2, max: 50 (sum = 150)"
    ) sequentialcovering(X, y, searchmethod=rs)

# === Reproducibility ======================================================================
# TODO Implementare confronto tra due DecisionList
