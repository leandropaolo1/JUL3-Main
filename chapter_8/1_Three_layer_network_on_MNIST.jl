using MLDatasets
using Random
Random.seed!(1)

Base.@kwdef mutable struct Number
    possible_matches::Int64=10
    number::Matrix{Float64}
    pixels::Int64=784
    target::Int8
end

Base.@kwdef mutable struct Layer
    node::Vector{Float64} = []
    hidden_size::Int64=40
    alpha::Float64=0.005
    error::Float64
    weights_0_1::Matrix{Float64} = 2 * rand(pixels, hidden_size) .- 1
    weights_1_2::Matrix{Float64} = 2 * rand(hidden_size, num_labels ) .- 1
end

Base.@kwdef mutable struct Node
    delta::Matrix{Float64} = zeros(1,4)
    pred::Matrix{Float64} = zeros(1,4)
end

function rectified_linear_unit(node::Node)
    node.pred = max.(node.pred, 0)
end

function rectified_linear_unit_derivative(node::Node)
    node.pred = node.pred .> 0
    pred = zeros(4,1)
    pred[:,1] = node.pred
    return pred
end

# testing 28 x 28 x 10,000

images = MNIST(:test).features
targets = MNIST(:test).targets
single_image = MNIST(:test).features[:, :, 1]
single_target = MNIST(:test).targets[1]

# training 28 x 28 x 60,000

training_images = MNIST(:train).features
training_targets = MNIST(:train).targets
training_single_image = MNIST(:train).features[:, :, 1]
training_single_target = MNIST(:train).targets[1]

for iter in 1:1000
    error, correct_cnt = (0.0, 0.0)

    for i in 1:1

    
    end
end