# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 125

using Random

Base.@kwdef mutable struct StreetLight
    targets::Matrix{Float64} = [0 1 0 0;]
    inputs = [
        1 0 1;
        0 1 1;
        0 0 1;
        1 1 1;]

end

Base.@kwdef mutable struct Layer
    iter::Int8 = 1
    alpha::Float64 = 0.2
    hidden_size::Int16 = 4
    pred::Matrix{Float64} = zeros(1,4)
    initial_layer::Matrix{Int64} = [1 0 1;]

    weights_0_1::Matrix{Float64} = 2 * rand(3, hidden_size) .- 1
    weights_0_2::Matrix{Float64} = 2 * rand(hidden_size, 1) .- 1
end

function relu(layer::Layer)
    return max.(layer.pred, 0)
end

layer_0 = Layer()
layer_1 = Layer().pred = layer_0.initial_layer * layer_0.weights_0_1
layer_1 = relu(layer_1)





    


    



