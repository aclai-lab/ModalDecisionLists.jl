using Test
using SoleBase: default_weights
using FillArrays
using StatsBase
using ModalDecisionLists
using ModalDecisionLists.Metrics: entropy, gini_impurity

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