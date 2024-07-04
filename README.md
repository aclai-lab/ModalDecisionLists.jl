# ModalDecisionLists.jl – it's synctactic sequential covering

[![Build Status](https://github.com/acla-lab/ModalDecisionLists.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/acla-lab/ModalDecisionLists.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Installation & Usage

Simply type the following commands in Julia's REPL:

```julia
# Install packages
using Pkg; Pkg.add("MLJ");
using Pkg; Pkg.add("ModalDecisionLists");

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
yhat = predict_mode(mach, rows = test_idxs)
accuracy = MLJ.accuracy(yhat, y[test_idxs])

# Access & inspect model
dlist = fitted_params(mach).fitresult.model
printmodel(dlist; show_metrics = true, show_subtree_metrics=true)

# Make test instances flow into the model
test_dlist = deepcopy(dlist)
apply!(test_dlist, slicedataset(PropositionalLogiset(X), test_idxs), y[test_idxs])

# Extract rules that perform well in test
interestingrules = listrules(test_dlist; min_lift = 1.0, min_coverage = 0.05, normalize = true)
printmodel.(interestingrules; show_metrics = (; round_digits = 2));
```


## Credits

*ModalDecisionLists.jl* lives within the [*Sole.jl*](https://github.com/aclai-lab/Sole.jl) framework for *symbolic machine learning*.
