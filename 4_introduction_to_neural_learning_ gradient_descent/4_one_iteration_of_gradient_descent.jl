# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 59

Base.@kwdef mutable struct GradDescent
    input::Float64 = 8.5
    weight::Float64 = 0.1
    target::Int128 = 10
    alpha::Float64 = 0.01
    pred::Float64 = 0.0
    error::Float64 = 0.0
end

function step!(grad::GradDescent)
    grad.pred = grad.input * grad.weight
    grad.error = round((grad.pred - grad.target) ^ 2, digits=3)
    grad.weight -= (((grad.pred - grad.target) * grad.input) * grad.alpha)
    println("Error: $(round(grad.error, digits=3)), Prediction: $(round(grad.pred, digits=3))")

end


grad_descent = GradDescent()

for iter in 1:10
    step!(grad_descent)
end

