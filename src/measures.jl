module Measures

using SoleBase: CLabel
using SoleBase: default_weights
using SoleModels
using FillArrays
using StatsBase
# TODO è come se diventasse un problema biclasse ?

function laplace_accuracy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
    n_labels::Integer = 2
)
    N = length(y)
    target_class = SoleModels.bestguess(y, suppress_parity_warning=true)
    n = countmap(y)[target_class]

    return -(n + 1) / (N + n_labels)
end

function entropy(
    y::AbstractVector{<:CLabel},
    w::AbstractVector=default_weights(length(y));
    kwargs...
)
    isempty(y) && return Inf

    distribution = (w isa Ones ? counts(y) : counts(y, Weights(w)))
    distribution = distribution[distribution .!= 0]
    # @show distribution
    length(distribution) == 1 && return 0.0

    prob = distribution ./ sum(distribution)
    e = -sum(prob .* log2.(prob))
    return e
end


end # module
