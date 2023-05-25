using MLDatasets
using Random
Random.seed!(1)

# testing 28 x 28 x 10,000

images = MNIST(:test).features
targets = MNIST(:test).targets
single_image = MNIST(:test).features[:, :, 1]
single_target = MNIST(:test).targets[1]

# training 28 x 28 x 60,000

images = MNIST(:train).features
targets = MNIST(:train).targets
single_image = MNIST(:train).features[:, :, 1]
single_target = MNIST(:train).targets[1]


function rectified_linear_unit(layer::Layer)
    layer.pred = max.(layer.pred, 0)
end

function rectified_linear_unit_derivative(layer::Layer)
    layer.pred = layer.pred .> 0
    pred = zeros(4,1)
    pred[:,1] = layer.pred
    return pred
end

Base.@kwdef mutable struct Number
    possible_matches::Int64=10
    pixels::Int64=784
    target::Int8=0

end

Base.@kwdef mutable struct Layer
    alpha::Float64=0.005
    hidden_size::Int64=40
    error::Float64 = 0.0
    pred::Matrix{Float64} = zeros(1,4)
    delta::Matrix{Float64} = zeros(1,4)
    weights_0_1::Matrix{Float64} = 2 * rand(pixels, hidden_size) .- 1
    weights_1_2::Matrix{Float64} = 2 * rand(hidden_size, num_labels ) .- 1
end
