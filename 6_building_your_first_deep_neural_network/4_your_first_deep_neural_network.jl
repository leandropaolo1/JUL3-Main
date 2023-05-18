# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 125
using Random

# `StreetLight` represents the input to the neural network.
Base.@kwdef mutable struct StreetLight
    # Targets represent the desired output for the corresponding inputs.
    targets::Matrix{Float64} = [0 1 0 0;]
    # Inputs represent the input data.
    inputs = [
        1 0 1;
        0 1 1;
        0 0 1;
        1 1 1;]
    # n_rows: number of rows in the input.
    n_rows = size(inputs,1)
    # n_cols: number of columns in the input.
    n_cols = size(inputs,2)
end

# `Layer` represents a layer in the neural network.
Base.@kwdef mutable struct Layer
    # Number of iterations for the network.
    iter::Int8 = 1
    # Learning rate.
    alpha::Float64 = 0.1
    # Size of the hidden layer.
    hidden_size::Int16 = 4
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

# Rectified Linear Unit (ReLU) activation function. 
# It applies the ReLU function elementwise to the predicted output.
function rectified_linear_unit(layer::Layer)
    layer.pred = max.(layer.pred, 0)
end

# Derivative of the Rectified Linear Unit (ReLU) activation function. 
# It calculates the derivative of the ReLU function applied to the predicted output.
function rectified_linear_unit_derivative(layer::Layer)
    layer.pred = layer.pred .> 0
    pred = zeros(4,1)
    pred[:,1] = layer.pred
    return pred
end

# Initialize the input and layers.
input = StreetLight()
layer_0 = Layer()
layer_1 = Layer()
layer_2 = Layer()

# Loop for 100 iterations (epochs).
for iter in 1:100
    layer_2.error = 0
    # Loop over each instance of input.
    for _iter in input.n_rows
        # Forward pass: Calculate the predicted output.
        layer_1.pred = input.inputs[_iter : _iter,:] * layer_0.weights_0_1
        rectified_linear_unit(layer_1)
        layer_2_pred = layer_1.pred * layer_0.weights_1_2
        layer_2.pred[1] = layer_2_pred[1]
        
        # Calculate the error (sum of squares of differences between predicted and target).
        layer_2.error += sum((layer_2.pred[1] .- input.targets[_iter:_iter]) .^ 2)
        
        # Backward pass: Calculate the gradient (delta).
        layer_2.delta[1] = layer_2.pred[1] - input.targets[_iter]
        layer_1.delta = layer_2.delta[1] * layer_0.weights_1_2 .* rectified_linear_unit_derivative(layer_1)
        
        # Update weights using gradient descent.
        layer_0.weights_1_2' .-= layer_0.alpha *layer_1.pred * layer_2.delta[1]
        layer_0.weights_0_1 .-= layer_0.alpha * input.inputs[_iter : _iter,:]' * layer_1.delta'
    end
    
    # Print the error every 10 iterations.
    if iter % 10 == 0
        println("Iteration: $iter -  Error: $(layer_2.error)")
    end 
end
