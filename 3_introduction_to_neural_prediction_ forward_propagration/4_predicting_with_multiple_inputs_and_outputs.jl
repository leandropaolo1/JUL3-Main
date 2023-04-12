# how do you take in oen value or inputs and output multiple prediction? Lets find out. This of course is a really simple neural network
# p.38 pdf.59
# Nice
"""
    predicting_with_multiple_inputs_and_outputs_1()

Predicting with multiple inputs and outputs
Neural networks can predict multiple outputs given
multiple inputs.
Finally, the way you build a network with multiple inputs or outputs can be combined to build
a network that has both multiple inputs and multiple outputs. As before, a weight connects each
input node to each output node, and prediction occurs in the usual way.

"""
function predicting_with_multiple_inputs_and_outputs_1()
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]

    weights = [
        0.1 0.1 -0.3;
        0.1 0.2 0.0;
        0.0 1.3 0.1]

    hurt_prediction = []
    win_prediction = []
    sad_prediction = []

    predictions = [hurt_prediction,win_prediction,sad_prediction]

    function w_sum(inputs, weights)
        array_product = inputs' * weights
        prediction = sum(array_product)
        return round(prediction, digits=3)
    end

    for iteration in eachindex(inputs[1,:])
        input_col = inputs[:,iteration]
        for second_iteration in eachindex(weights[:,1])
            prediction = w_sum(input_col, weights[second_iteration,:])
            append!(predictions[second_iteration],prediction)
        end
    end

    println("Hurt Prediction: ",predictions[1])
    println("Win Prediction : ",predictions[2])
    println("Sad Prediction : ",predictions[3])

end

"""
    predicting_with_multiple_inputs_and_outputs_2()

Predicting with multiple inputs and outputs
Neural networks can predict multiple outputs given
multiple inputs.
Finally, the way you build a network with multiple inputs or outputs can be combined to build
a network that has both multiple inputs and multiple outputs. As before, a weight connects each
input node to each output node, and prediction occurs in the usual way.

"""
function predicting_with_multiple_inputs_and_outputs_2()
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]

    weights = [
        0.1 0.1 -0.3;
        0.1 0.2 0.0;
        0.0 1.3 0.1]

    n_rows = size(inputs,1)
    n_cols = size(inputs,2)

    prediction = zeros(n_rows, n_cols)

    for iter in 1:n_cols
        col = inputs[:,iter]
        prediction[:,iter]= round.(weights * col, digits=3)

    end

    return prediction

end


begin
    predicting_with_multiple_inputs_and_outputs_2()
end

