using Test
using SoleBase: default_weights, CLabel
using FillArrays
using StatsBase
using ModalDecisionLists
using ModalDecisionLists: Antecedent
using ModalDecisionLists.Metrics: entropy, gini_impurity
using RDatasets
using BenchmarkTools

@testset "Entropy Tests" begin
    # Test 1: Empty vector should return Inf
    @test entropy(UInt32[], default_weights(0); nlabels=2) == Inf
    
    # Test 2: Single class (all same labels) should return 0.0
    y_single = UInt32[1, 1, 1, 1]
    @test entropy(y_single; nlabels=2) == 0.0
    
    # Test 3: Uniform binary distribution
    # For uniform distribution of 2 classes: entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
    y_uniform_binary = UInt32[1, 2, 1, 2]
    @test isapprox(entropy(y_uniform_binary; nlabels=2), 1.0, atol=1e-10)
    
    # Test 4: Uniform ternary distribution
    # For uniform distribution of 3 classes: entropy = -3 * (1/3)*log2(1/3) ≈ 1.585
    y_uniform_ternary = UInt32[1, 2, 3, 1, 2, 3]
    expected_entropy = -3 * (1/3) * log2(1/3)
    @test isapprox(entropy(y_uniform_ternary; nlabels=3), expected_entropy, atol=1e-10)
    
    # Test 5: Weighted entropy
    y_weighted = UInt32[1, 2, 1, 2]
    w_weighted = [0.25, 0.75, 0.25, 0.75]  # More weight on class 2
    entropy_weighted = entropy(y_weighted, w_weighted; nlabels=2)
    # Distribution becomes [0.5, 1.5], normalized: [0.25, 0.75]
    expected = -(0.25 * log2(0.25) + 0.75 * log2(0.75))
    @test isapprox(entropy_weighted, expected, atol=1e-10)
    
    # Test 6: Negative nlabels should throw ArgumentError
    y = UInt32[1, 2, 1, 2]
    @test_throws ArgumentError entropy(y; nlabels=-1)
    
    # Test 7: Non-uniform distribution
    y_nonuniform = UInt32[1, 1, 1, 2]  # 3 of class 1, 1 of class 2
    probs = [0.75, 0.25]
    expected = -(0.75 * log2(0.75) + 0.25 * log2(0.25))
    @test isapprox(entropy(y_nonuniform; nlabels=2), expected, atol=1e-10)
end

@testset "Gini Impurity Tests" begin
    # Test 1: Empty vector should return Inf
    @test gini_impurity(UInt32[], default_weights(0); nlabels=2) == Inf
    
    # Test 2: Single class (pure) should return 0.0
    y_single = UInt32[1, 1, 1, 1]
    @test gini_impurity(y_single; nlabels=2) == 0.0
    
    # Test 3: Uniform binary distribution
    # For uniform distribution: gini = 1 - (0.5^2 + 0.5^2) = 0.5
    y_uniform_binary = UInt32[1, 2, 1, 2]
    @test isapprox(gini_impurity(y_uniform_binary; nlabels=2), 0.5, atol=1e-10)
    
    # Test 4: Non-uniform binary distribution
    # For [0.75, 0.25]: gini = 1 - (0.75^2 + 0.25^2) = 0.375
    y_nonuniform_binary = UInt32[1, 1, 1, 2]
    expected_gini = 1 - (0.75^2 + 0.25^2)
    @test isapprox(gini_impurity(y_nonuniform_binary; nlabels=2), expected_gini, atol=1e-10)
    
    # Test 5: Uniform ternary distribution
    # For [1/3, 1/3, 1/3]: gini = 1 - 3*(1/3)^2 = 2/3
    y_uniform_ternary = UInt32[1, 2, 3]
    expected_gini = 1 - 3 * (1/3)^2
    @test isapprox(gini_impurity(y_uniform_ternary; nlabels=3), expected_gini, atol=1e-10)
    
    # Test 6: Weighted gini impurity
    y_weighted = UInt32[1, 2, 1, 2]
    w_weighted = [0.25, 0.75, 0.25, 0.75]  # More weight on class 2
    gini_weighted = gini_impurity(y_weighted, w_weighted; nlabels=2)
    # Distribution becomes [0.5, 1.5], normalized: [0.25, 0.75]
    expected = 1 - (0.25^2 + 0.75^2)
    @test isapprox(gini_weighted, expected, atol=1e-10)
    
    # Test 7: Negative nlabels should throw ArgumentError
    y = UInt32[1, 2, 1, 2]
    @test_throws ArgumentError gini_impurity(y; nlabels=-1)
    
    # Test 8: Zero nlabels should throw ArgumentError
    @test_throws ArgumentError gini_impurity(y; nlabels=0)
end

# @testset "FOILGain" begin
    # small dummy data  
# y = convert(UInt32, rand([1, 2], 1e6))
# w = default_weights(length(y))
# target = 1

# # instantiate the loss functor
# lossf = ModalDecisionLists.LossFunctions.FOILGain()

# # missing antecedents should yield zero loss
# # @test lossf(y, w, target) == 0
# # @test lossf(y, w, target; antecedent=nothing) == 0
# # @test lossf(y, w, target; prev_antecedent=nothing) == 0

# # two handcrafted antecedents with simple coverage masks
# # ant1 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true, false, true, false]))
# ant0_coverage = rand([0, 1], 1e6)
# ant1_coverage = rand([0, 1], 1e6)
# ant0 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector(ant0_coverage))
# ant1 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector(ant1_coverage))

# @btest lossf(y, w, target; antecedent=ant1, prev_antecedent=ant0)

    # compute expected value with same logic as implementation
    # tmask = (y .== target) .> 0
    # tp1 = sum((ant1.covmask .& tmask) .* w)
    # fp1 = sum((ant1.covmask .& .!tmask) .* w)
    # tp0 = sum((ant0.covmask .& tmask) .* w)
    # fp0 = sum((ant0.covmask .& .!tmask) .* w)
    # prec_curr = (tp1 + fp1 > 0) ? tp1 / (tp1 + fp1) : 0.0
    # prec_prev = (tp0 + fp0 > 0) ? tp0 / (tp0 + fp0) : 0.0
    # t = sum((ant0.covmask .& ant1.covmask) .* w)
    # expected = -t * (log2(prec_curr) - log2(prec_prev))

    # @test isapprox(lossf(y, w, target; antecedent=ant1, prev_antecedent=ant0),
    #                expected, atol=1e-12)
# end


@testset "LaplaceAccuracy Tests" begin
    # Test 1: All samples are target class (perfect precision)
    # Expected: 1 - (n + 1)/(n + 0 + 2) → approaches 0 as n grows
    y_all_target = UInt32[1, 1, 1, 1]
    w_uniform = default_weights(4)
    target_class = UInt32(1)
    nlabels = 2
    
    # Create a mock antecedent that covers all samples
    mock_ant = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true, true, true, true]))
    lossf = ModalDecisionLists.LossFunctions.LaplaceAccuracy()
    loss = lossf(y_all_target, w_uniform, target_class; antecedent=mock_ant, nlabels=nlabels)
    expected = 1 - (4 + 1) / (4 + 0 + 2)  # = 1 - 5/6 ≈ 0.1667
    @test isapprox(loss, expected, atol=1e-10)
    
    # Test 2: No samples are target class (zero precision)
    # Expected: 1 - (0 + 1)/(n + 2) → approaches 1 as n grows
    y_no_target = UInt32[2, 2, 2, 2]
    mock_ant2 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true, true, true, true]))
    loss = lossf(y_no_target, w_uniform, target_class; antecedent=mock_ant2, nlabels=nlabels)
    expected = 1 - (0 + 1) / (4 + 2)  # = 1 - 1/6 ≈ 0.8333
    @test isapprox(loss, expected, atol=1e-10)
    
    # Test 3: Mixed distribution (50% target, 50% non-target)
    y_mixed = UInt32[1, 2, 1, 2]
    mock_ant3 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true, true, true, true]))
    loss = lossf(y_mixed, w_uniform, target_class; antecedent=mock_ant3, nlabels=nlabels)
    expected = 1 - (2 + 1) / (4 + 2)  # = 1 - 3/6 = 0.5
    @test isapprox(loss, expected, atol=1e-10)
    
    # Test 4: Partial coverage - only some samples covered by antecedent
    y_partial = UInt32[1, 1, 2, 2]
    mock_ant4 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true, true, false, false]))
    loss = lossf(y_partial, w_uniform, target_class; antecedent=mock_ant4, nlabels=nlabels)
    expected = 1 - (2 + 1) / (2 + 2)  # = 1 - 3/4 = 0.25
    @test isapprox(loss, expected, atol=1e-10)
    
    # Test 5: Weighted samples
    y_weighted = UInt32[1, 1, 2, 2]
    w_weighted = [0.25, 0.25, 0.5, 0.5]  # More weight on negative class
    mock_ant5 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true, true, true, true]))
    loss = lossf(y_weighted, w_weighted, target_class; antecedent=mock_ant5, nlabels=nlabels)
    tp = 0.25 + 0.25  # weight of target class samples
    fp = 0.5 + 0.5    # weight of non-target class samples
    expected = 1 - (tp + 1) / (tp + fp + 2)
    @test isapprox(loss, expected, atol=1e-10)
    
    # Test 6: No coverage (antecedent covers no samples)
    y_test = UInt32[1, 2, 1, 2]
    mock_ant6 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([false, false, false, false]))
    loss = lossf(y_test, w_uniform, target_class; antecedent=mock_ant6, nlabels=nlabels)
    expected = 1 - (0 + 1) / (0 + 2)  # = 1 - 1/2 = 0.5 (Laplace smoothing effect)
    @test isapprox(loss, expected, atol=1e-10)
    
    # Test 7: Single sample covered - Laplace smoothing is critical
    y_single = UInt32[1]
    w_single = [1.0]
    mock_ant7 = Antecedent(LeftmostConjunctiveForm([⊤]), BitVector([true]))
    loss = lossf(y_single, w_single, target_class; antecedent=mock_ant7, nlabels=nlabels)
    expected = 1 - (1 + 1) / (1 + 0 + 2)  # = 1 - 2/3 ≈ 0.3333
    @test isapprox(loss, expected, atol=1e-10)
end