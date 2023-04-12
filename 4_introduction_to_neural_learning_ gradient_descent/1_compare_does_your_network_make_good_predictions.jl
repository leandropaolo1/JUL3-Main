"""
    compare_does_your_neural_network_make_good_predictions()

Compare: Does your network make
good predictions?
Let’s measure the error and find out!
Execute the following code in your Jupyter notebook. It should print 0.3025:
"""
function compare_does_your_neural_network_make_good_predictions()

    knob_weight = 0.5
    input = 0.5
    target = 0.8

    prediction = input * knob_weight
    error = (prediction - target) ^ 2

    
end
