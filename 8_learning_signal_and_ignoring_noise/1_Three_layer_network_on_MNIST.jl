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


function rectified_linear_unit(number::Number)

end

function rectified_linear_unit_derivative(number::Number)

end

Base.@kwdef mutable struct Number
    possible_matches::Int64=10
    hidden_size::Int64=40
    alpha::Float64=0.005
    pixels::Int64=784
    target::Int8
    pred::Matrix{Float64} = zeros(1,hidden_size)
    delta::Matrix{Float64} = zeros(1,hidden_size)
    weights_0_1::Matrix{Float64} = 2 * rand(pixels, hidden_size) .- 1
    weights_1_2::Matrix{Float64} = 2 * rand(hidden_size, num_labels ) .- 1
end

Base.@kwdef mutable struct Layer
    # Learning rate.
    alpha::Float64 = 0.1
    # Size of the hidden layer.
    hidden_size::Int16 = 40
    # Error in prediction.
    error::Float64 = 0.0
    # Predicted output.
    pred::Matrix{Float64} = zeros(1,4)
    # Delta represents the gradient or derivative of the error.
    delta::Matrix{Float64} = zeros(1,4)
    # Weights for the connections between the input layer and the hidden layer.
    weights_0_1::Matrix{Float64} = 2 * rand(3, hidden_size) .- 1
    # Weights for the connections between the hidden layer and the output layer.
    weights_1_2::Matrix{Float64} = 2 * rand(hidden_size, 1) .- 1
end
