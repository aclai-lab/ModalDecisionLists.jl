using Test
using ModalDecisionLists.LossFunctions: significance_test

function createsampling(dist)
    vettore = Int[]
    for (val, ripetizioni) in enumerate(dist)
        append!(vettore, fill(val, ripetizioni))
    end
    return vettore
end

a = createsampling([11  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([16  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1; n_labels=3) == true

a = createsampling([23  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([37  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([44  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([48  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([50  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([50  0  0])
b = createsampling([50 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([ 0 50 50])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([0 3 1])
b = createsampling([0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([0 4 1])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([ 0 47 49])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([0 5 1])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([ 0 46 49])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([0 6 1])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([ 0 45 49])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == false

a = createsampling([ 0 11  1])
b = createsampling([ 0 50 50])
@test significance_test(a,b, 0.1, n_labels=3) == true

a = createsampling([ 0 11  1])
b = createsampling([ 0 50 50])
@test_nowarn significance_test(a,b, 0.0, n_labels=3)
@test_nowarn significance_test(a,b, 1.0, n_labels=3)
