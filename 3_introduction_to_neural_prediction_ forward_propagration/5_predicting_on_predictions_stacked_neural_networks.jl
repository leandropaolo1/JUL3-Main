    # how do you take in oen value or inputs and output multiple prediction? Lets find out. This of course is a really simple neural network
# p.42 pdf.63
# Nice

"""
    predicting_on_predictions_stacked_neural_networks_1()

Predicting on predictions
Neural networks can be stacked!
As the following figures make clear, you can also take the output of one network and feed it
as input to another network. This results in two consecutive vector-matrix multiplications.
It may not yet be clear why you’d predict this way; but some datasets (such as image
classification) contain patterns that are too complex for a single-weight matrix. Later, we’ll
discuss the nature of these patterns. For now, it’s sufficient to know this is possible.
"""
function predicting_on_predictions_stacked_neural_networks_1()
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]

    weights1 = [
        0.1 0.1 -0.3;
        0.1 0.2 0.0;
        0.0 1.3 0.1]
    
    weights2 = [
        0.3 1.1 -0.3;
        0.1 0.2 0.0;
        0.0 1.3 0.1]

    combined_weights = [weights1, weights2]


    function w_sum(inputs, weights)
        array_product = inputs' * weights
        prediction = sum(array_product)
        return round(prediction, digits=3)
    end

    function m_mul(inputs, weights)
        predictions = [[],[],[]]

        for iteration in eachindex(inputs[1,:])
            input_col = inputs[:,iteration]
            for second_iteration in eachindex(weights[:,1])
                prediction = w_sum(input_col, weights[second_iteration,:])
                append!(predictions[second_iteration],prediction)
            end
        end

        return predictions


    end

    function m_mul_formatted(inputs, weights)
        n_rows = size(weights, 1)
        n_cols = size(inputs, 2)
        predictions = zeros(n_rows, n_cols)
    
        for i in 1:n_cols
            col = inputs[:, i]
            predictions[:, i] = round.(weights * col, digits=3)
        end


        return predictions
    end
    col = inputs[:,1]
    println(round.(weights1 * col, digits=3))

    hiddens = m_mul_formatted(inputs,weights1)
    predictions = m_mul_formatted(hiddens,weights2)

end


"""
    predicting_on_predictions_stacked_neural_networks_2()

Predicting on predictions
Neural networks can be stacked!
As the following figures make clear, you can also take the output of one network and feed it
as input to another network. This results in two consecutive vector-matrix multiplications.
It may not yet be clear why you’d predict this way; but some datasets (such as image
classification) contain patterns that are too complex for a single-weight matrix. Later, we’ll
discuss the nature of these patterns. For now, it’s sufficient to know this is possible.
"""
function predicting_on_predictions_stacked_neural_networks_2()
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
    predicting_on_predictions_stacked_neural_networks_2()
end