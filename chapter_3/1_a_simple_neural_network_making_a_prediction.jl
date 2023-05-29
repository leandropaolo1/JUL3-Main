# so what is the simplest neural network available? Lets find out!


"""
    Simple neural network.

# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 24
Given an input and a weight, compute a prediction using multiplication.

# Author
- `Leandro Cooper`

# Arguments
- `input::Float64`: a single input value.
- `weight::Float64`: a single weight value.

# Returns
- `prediction::Float64`: the prediction value.

# Example
```julia
julia> input = 0.5
julia> weight = 0.2
julia> prediction = simple_neural_network(input, weight)
julia> println(prediction)
0.1


"""
function simple_neural_network(input, weight)
    prediction = input * weight
    return prediction
end

begin

    input = 8.5
    weight = 0.1
    simple_neural_network(input,weight)
end