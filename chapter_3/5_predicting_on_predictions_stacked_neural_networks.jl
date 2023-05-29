"""
predicting_on_predictions_stacked_neural_networks()


# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 42
This function implements a stacked neural network that makes predictions based on predictions
from a previous layer. It takes in three input vectors representing the number of toes, the weight
of the team, and the number of fans, and then makes predictions using two layers of neural networks.

# Arguments
- `toes`: a vector of numbers representing the number of toes.
- `wlrec`: a vector of numbers representing the weight of the team.
- `nfans`: a vector of numbers representing the number of fans.

# Returns
- `first_layer`: a matrix of predictions from the first layer of the neural network.
- `second_layer`: a matrix of predictions from the second layer of the neural network.

# Example
toes = [8.5 9.5 9.9 9.0]
wlrec = [0.65 0.8 0.8 0.9]
nfans = [1.2 1.3 0.5 1.0]
(first_layer, second_layer) = predicting_on_predictions_stacked_neural_networks()

"""
function predicting_on_predictions_stacked_neural_networks()
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]

    weights1= [
        0.1 0.2 -0.1;
        -0.1 0.1 0.9;
        0.1 0.4 0.1] 

    weights2 = [
        0.3 1.1 -0.3;
        0.1 0.2 0.0;
        0.0 1.3 0.1;] 

    function predictor(inputs, weights)
        n_rows = size(weights,1)
        n_cols = size(inputs,2)

        predictions = zeros(n_rows, n_cols)

        for iter in 1:n_cols
            col = inputs[:,iter]
            predictions[:,iter] = round.(weights*col, digits=3)
        end

        return predictions
    
    end

    first_layer = predictor(inputs, weights1)
    second_layer = predictor(first_layer, weights2)

    return (first_layer, second_layer)

end

begin
    predicting_on_predictions_stacked_neural_networks()
end