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
    hidden_size::Int64=40
    alpha::Float64=0.005
    num_labels::Int64=10
    pixels::Int64=784
    target::Int8
    pred::Matrix{Float64} = zeros(1,hidden_size)
    delta::Matrix{Float64} = zeros(1,hidden_size)
    weights_0_1::Matrix{Float64} = 2 * rand(pixels, hidden_size) .- 1
    weights_1_2::Matrix{Float64} = 2 * rand(hidden_size, num_labels ) .- 1
end