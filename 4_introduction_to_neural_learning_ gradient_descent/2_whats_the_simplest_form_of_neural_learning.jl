"""
    Learning using the hot and cold method.
At the end of the day, learning is really about one thing: adjusting knob_weight either up
or down so the error is reduced. If you keep doing this and the error goes to 0, you’re done
learning! How do you know whether to turn the knob up or down? Well, you try both up and
down and see which one reduces the error! Whichever one reduces the error is used to update
knob_weight. It’s simple but effective. After you do this over and over again, eventually
error == 0, which means the neural network is predicting with perfect accuracy.
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