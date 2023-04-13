# how do you take in multiple values or inputs and output a prediction? Lets find out. This of course is a really simple neural network
# p.28 pdf.49

"""
    making_a_prediction_with_multiple_inputs()

# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 24
Returns a prediction based on the weighted sum of inputs using the formula:

    prediction = weights * inputs

where `weights` is a vector of weights and `inputs` is a matrix with each column representing a set of inputs.
# Author
- `Leandro Cooper`

# Arguments
- `toes::Array{Float64,2}`: An array of floats representing the number of toes a player has.
- `wlrec::Array{Float64,2}`: An array of floats representing the win/loss record of the player's team.
- `nfans::Array{Float64,2}`: An array of floats representing the number of fans in attendance.
- `weights::Array{Float64,2}`: An array of floats representing the weights for each input.

# Output
- `prediction::Array{Float64,2}`: An array of floats representing the prediction for each set of inputs.

# Examples
```julia
julia> making_a_prediction_with_multiple_inputs()
2×4 Array{Float64,2}:
 0.98  1.15  1.14  1.26
 0.56  0.77  0.57  0.87

"""
function making_a_prediction_with_multiple_inputs()
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]
    weights = [0.1 0.2 0]

    n_rows = size(weights,2) == 1 ? 1 : 1
    n_cols = size(inputs,2)

    prediction = zeros(n_rows, n_cols)
    
    for iter in 1:n_cols
        col = inputs[:,iter]
        prediction[:,iter] = round.(weights * col, digits=3)        
    end
    
    return prediction
end 


begin
    making_a_prediction_with_multiple_inputs()
end