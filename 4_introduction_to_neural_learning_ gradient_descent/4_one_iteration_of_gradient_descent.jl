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

function step!(l::GradDescent)
    l.pred = l.input * l.weight
    l.error = round((l.pred - l.target) ^ 2, digits=3)
    l.weight -= (l.input * (l.pred - l.target)) * l.alpha
    println("Error: $(round(l.pred, digits=3)), Prediction: $(l.error)")
end

grad_descent = GradDescent()

for iter in 1:10
    step!(grad_descent)
end