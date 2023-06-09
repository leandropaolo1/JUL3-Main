# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 90

Base.@kwdef mutable struct GradDescent
    alpha::Float64 = 0.01
    error::Float64 = 0.0
    delta::Float64 = 0.0
    pred::Float64 = 0.0

    iter::Int16 = 1
    target = [1 1 0 1]
    weights = [0.1 0.2 -0.1]

    inputs = [
        8.5 9.5 9.9 9.0;
        0.65 0.8 0.8 0.9;
        1.2 1.3 0.5 1.0]
    
    n_cols = size(inputs, 2)
    n_rows = size(weights, 2) == 1 ? 1 : size(weights, 1)
    weighted_deltas = zeros(1, 3)

end


function step!(grad::GradDescent, col::Int64)
    for _ in 1:10
        column = grad.inputs[:,col]
        grad.pred = column' .* grad.weights

    end

end