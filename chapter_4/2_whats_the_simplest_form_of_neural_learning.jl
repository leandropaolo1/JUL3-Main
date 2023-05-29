

"""
    whats_the_simplest_form_of_neural_learning()


# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 52
A simple neural learning algorithm to predict the value of the target by tuning the weight of the input. 

# Arguments
- `input::Float64`: input value
- `weight::Float64`: initial weight value
- `target::Float64`: target value to predict
- `step_amount::Float64`: amount by which weight is adjusted at each iteration

# Returns
- `Nothing`: the function doesn't return anything, it simply prints the error and prediction at each iteration to the console

# Examples
```julia
julia> whats_the_simplest_form_of_neural_learning()
Error: 0.04000000000000007 Prediction: 0.05
Error: 0.038400000000000034 Prediction: 0.060000000000000005
Error: 0.03686400000000003 Prediction: 0.06999999999999999
Error: 0.03539136000000002 Prediction: 0.08
Error: 0.03398083840000002 Prediction: 0.09
...
"""
function whats_the_simplest_form_of_neural_learning()
    input = 0.5
    weight = 0.1

    target = 0.8

    step_amount = 0.01

    for iter in 1:1101
        prediction = input * weight
        error = (prediction - target) ^ 2

        println("Error: $error Prediction: $prediction")
        up_prediction = input * (weight + step_amount)
        up_error = (target - up_prediction) ^ 2

        down_prediction = input * (weight - step_amount)
        down_error = (target - down_prediction) ^ 2

        if down_error < up_error
            weight = weight - step_amount
        
        end
        if down_error > up_error
            weight = weight + step_amount
        end

    end
end


begin
    whats_the_simplest_form_of_neural_learning()
end