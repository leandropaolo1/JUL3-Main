# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 52

function calculating_both_direction_and_amount_from_error()
    weight = 0.5
    target = 0.8
    input = 0.5
    for iter in 1:400
        pred = input * weight
        error = (pred - target) ^ 2
        direction_and_amount = (pred - target) * input
        weight -= direction_and_amount
        
        println("Error: $error Prediction: $pred")
    end
end


begin
    calculating_both_direction_and_amount_from_error()
end


