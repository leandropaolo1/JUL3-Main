
function calculating_both_direction_and_amount_from_error()
    weight = 0.5
    target = 0.8
    input = 0.5
    for iter in 1:300
        pred = input * weight
        error = (pred - target) ^ 2
        direction_and_amount = (pred - target) * input
        weight -= direction_and_amount
        
        println("Error: $error Prediction: $pred")
    end
end

function calculating_both_direction_and_amount_from_error()
    target = 0.8
    weight =  0.5
    input = 0.5

    for iter in 1:300
        pred = input * weight
        error = (pred - target) ^ 2
        direction = (pred - target) * input
        weight -= direction

end

begin
    calculating_both_direction_and_amount_from_error()
end


