using Test
using ModalDecisionLists

using MLJ
using DataFrames
using Random

# Load an example dataset
X, y = MLJ.@load_iris()
X = DataFrame(X)
rng = Random.Xoshiro(42)

# Split dataset
train_ratio = 0.7
train, test = MLJ.partition(eachindex(y), train_ratio; shuffle=true, rng)
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]

# ---------------------------------------------------------------------------- #
#                          decision tree classifier                            #
# ---------------------------------------------------------------------------- #
# Instantiate an MLJ machine
model = DecisionListClassifier(; rng)
mach = machine(model, X, y)

# Fit the model
MLJ.fit!(mach; rows=train, verbosity=0)

# Perform predictions, compute accuracy
yhat = @test_nowarn predict(mach, rows=test)
accuracy = MLJ.accuracy(yhat, y_test)

# Access & inspect model
dlist = @test_nowarn fitted_params(mach).fitresult.model
@test_nowarn printmodel(dlist; show_metrics = true, show_subtree_metrics=true)

@test_nowarn apply(dlist, slicedataset(PropositionalLogiset(X), test))

# @test_nowarn apply!(test_dlist, slicedataset(PropositionalLogiset(X), test), y_test)
# @test_nowarn printmodel(test_dlist; show_metrics = true, show_subtree_metrics=true)
# @test_nowarn apply!(test_dlist, slicedataset(PropositionalLogiset(X), test), y_test; mode=:append, show_progress=true)

@test_nowarn listrules(dlist)
printmodel.(listrules(dlist); show_metrics = true, show_subtree_metrics=true);

readmetrics.(listrules(dlist))
printmodel.(listrules(dlist, normalize=true); show_metrics = (; round_digits=nothing));

# ---------------------------------------------------------------------------- #
#                      random decision tree classifier                         #
# ---------------------------------------------------------------------------- #
# Instantiate an MLJ machine
model = RandomDecisionListClassifier(; rng)
mach = machine(model, X, y)

# Fit the model
MLJ.fit!(mach; rows=train, verbosity=0)

# Perform predictions, compute accuracy
yhat = @test_nowarn predict(mach, rows=test)
accuracy = MLJ.accuracy(yhat, y_test)

# Access & inspect model
dlist = @test_nowarn fitted_params(mach).fitresult.model
@test_nowarn printmodel(dlist; show_metrics = true, show_subtree_metrics=true)

@test_nowarn apply(dlist, slicedataset(PropositionalLogiset(X), test))

# @test_nowarn apply!(test_dlist, slicedataset(PropositionalLogiset(X), test), y_test)
# @test_nowarn printmodel(test_dlist; show_metrics = true, show_subtree_metrics=true)
# @test_nowarn apply!(test_dlist, slicedataset(PropositionalLogiset(X), test), y_test; mode=:append, show_progress=true)

# @test_nowarn listrules(dlist)
# printmodel.(listrules(dlist); show_metrics = true, show_subtree_metrics=true);

# readmetrics.(listrules(dlist))
# printmodel.(listrules(dlist, normalize=true); show_metrics = (; round_digits=nothing));
