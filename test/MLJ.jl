using Test

# Import packages
using MLJ
using ModalDecisionLists
using Random

# Load an example dataset
X, y = MLJ.@load_iris()
N = length(y)

# Instantiate an MLJ machine
mach = machine(ExtendedSequentialCovering(), X, y)

# Split dataset
p = randperm(N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
yhat = @test_nowarn predict(mach, rows=test_idxs)
accuracy = MLJ.accuracy(yhat, y[test_idxs])

# Access & inspect model
dlist = @test_nowarn fitted_params(mach).fitresult.model
@test_nowarn printmodel(dlist; show_metrics = true, show_subtree_metrics=true)

@test_nowarn apply(dlist, slicedataset(PropositionalLogiset(X), test_idxs))

# Make instances flow into the model
test_dlist = deepcopy(dlist)
@test_nowarn apply!(test_dlist, slicedataset(PropositionalLogiset(X), test_idxs), y[test_idxs])
printmodel(test_dlist; show_metrics = true, show_subtree_metrics=true)

@test_nowarn apply!(test_dlist, slicedataset(PropositionalLogiset(X), test_idxs), y[test_idxs]; mode=:append, show_progress=true)

@test_nowarn listrules(test_dlist)

printmodel.(listrules(test_dlist); show_metrics = true, show_subtree_metrics=true);

readmetrics.(listrules(test_dlist))

printmodel.(listrules(test_dlist, normalize = true); show_metrics = (; round_digits = nothing));

@test listrules(test_dlist; min_lift = 1.0, min_coverage = 0.05, normalize = true)

@test_nowarn listrules(test_dlist; min_lift = 1.0, min_coverage = 0.05)

interesting_rules = @test_nowarn SoleModels.listrules(test_dlist; use_shortforms=true, min_confidence=0.2, min_lift=1.0,
       min_ninstances = 2, custom_filter_callback = (ms)->ms.coverage*ms.ninstances > 1)

@test_nowarn printmodel.(sort(interesting_rules, by = x->readmetrics(x).confidence, rev = true); show_metrics = (; round_digits = nothing));

