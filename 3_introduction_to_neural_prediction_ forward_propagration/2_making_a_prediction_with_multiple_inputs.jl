# how do you take in multiple values or inputs and output a prediction? Lets find out. This of course is a really simple neural network
# p.28 pdf.49

function making_a_prediction_with_multiple_inputs_1()
    function w_sum(inputs, weights)
        array_product = inputs' * weights
        prediction = sum(array_product)
        return prediction
    end
    results = []
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]
    weights = [0.1, 0.2, 0]

    for col=1:length(toes[1:end])
        prediction = w_sum(inputs[:,col],weights)
        append!(results,round(prediction, digits=3))
    end

    return results
end

function making_a_prediction_with_multiple_inputs_2()
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
    making_a_prediction_with_multiple_inputs_2()
end