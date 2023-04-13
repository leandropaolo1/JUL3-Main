# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 59


function one_iteration_of_gradient_descent()
    input = 8.5
    weight = 0.1
    target = 1.0
    alpha = 0.01

    for iter in 1:10
        pred = input * weight
        error = (pred - target) ^ 2
        delta = pred - target
        weighted_delta = input * delta
        weight -= weighted_delta * alpha

        println("Iter: $iter Error: $(round(error, digits=5)) Prediction: $(round(pred, digits=5))")

    end

end

function one_iteration_of_gradient_descent()
    input = 8.5
    weight = 0.1
    target = 1.0
    alpha = 0.01

    for iter in 1:10
        pred = input * weight
        error = (pred - target) ^ 2
        weight =- (input * (pred-target)) * alpha

        println("Iter: $iter Error: $(round(error, digits=5)) Prediction: $(round(pred, digits=5))")

    end
    
end

begin
    one_iteration_of_gradient_descent()
end