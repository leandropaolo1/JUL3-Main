"""
    compare_does_your_neural_network_make_good_predictions()

# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 50
    
Compute the prediction of a neural network given an input value and a knob weight, 
and calculate the error between the prediction and the target value.

# Arguments
- `knob_weight::Float64`: a number between 0 and 1 representing the weight of the knob.
- `input::Float64`: a number between 0 and 1 representing the input value.
- `target::Float64`: a number between 0 and 1 representing the target value.

# Returns
- `error::Float64`: the squared error between the prediction and the target value.

# Example
```julia
julia> compare_does_your_neural_network_make_good_predictions()
0.09

"""
function compare_does_your_neural_network_make_good_predictions()

    knob_weight = 0.5
    input = 0.5
    target = 0.8

    prediction = input * knob_weight
    error = (prediction - target) ^ 2

end

begin
    compare_does_your_neural_network_make_good_predictions()
end
