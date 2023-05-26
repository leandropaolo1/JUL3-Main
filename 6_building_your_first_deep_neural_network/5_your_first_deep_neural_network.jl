using Random

Base.@kwdef mutable struct StreetLight
    inputs::Matrix{Int8} = [1 0 1; 0 1 1; 0 0 1; 1 1 1;]
    targets::Matrix{Int8} = [0 1 0 0;]
    n_rows = size(inputs, 1)
    n_cols = size(inputs, 2)
end

Base.@kwdef mutable struct Layer
    alpha::Float64 = 0.1
    error::Float64 = 0.0
    hidden_nodes::Int16 = 4
    pred::Matrix{Float64} = zeros(1, hidden_nodes)
    delta::Matrix{Float64} = zeros(1, hidden_nodes)
    weights_0_1::Matrix{Float64} = 2 * rand(3, hidden_nodes) .- 1
    weights_1_2::Matrix{Float64} = 2 * rand(hidden_nodes, 1) .- 1
end

function rectified_linear_unit(layer::Layer)
    layer.pred = max.(layer.pred, 0)
end

function rectified_linear_unit_derivative(layer::Layer)
    layer.pred = layer.pred .> 0
    pred = zeros(4, 1)
    pred[:, 1] = layer.pred
    return pred
end

input = StreetLight()
layer_0 = Layer()
layer_1 = Layer()
layer_2 = Layer()

for iter in 1:100
    layer_2.error = 0
    for _iter in input.n_rows
        layer_1.pred = input.inputs[_iter:_iter, :] * layer_0.weights_0_1
        rectified_linear_unit(layer_1)
        layer_2_pred = layer_1.pred * layer_0.weights_1_2
        layer_2.pred[1] = layer_2_pred[1]
        layer_2.error += sum((layer_2.pred[1] .- input.targets[_iter:_iter]) .^ 2)
        layer_2.delta[1] = layer_2.pred[1] - input.targets[_iter]
        layer_1.delta = layer_2.delta[1] * layer_0.weights_1_2 .* rectified_linear_unit_derivative(layer_1)
        layer_0.weights_1_2' .-= layer_0.alpha * layer_1.pred * layer_2.delta[1]
        layer_0.weights_0_1 .-= layer_0.alpha * input.inputs[_iter:_iter, :]' * layer_1.delta'
    end

    if iter % 10 == 0
        println("Iteration: $iter - Error: $(layer_2.error)")
    end
end
