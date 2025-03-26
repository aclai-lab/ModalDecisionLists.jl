using ModalDecisionLists
using Test
using Random

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        @time include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("Core", ["sequentialcovering.jl",]),
    ("CN2 correctness", ["cn2-correctness.jl"]),
    ("MLJ", ["MLJ.jl", "cn2-accuracy-MLJ.jl"]),
    ("CN2 benchmark", ["cn2-benchmark.jl"]),
    ("Orange", ["orange-decisionlists.jl"]),

]

@testset "ModalDecisionLists.jl" begin
    for ts in eachindex(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
    println()
end
