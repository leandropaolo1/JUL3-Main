# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 80

Base.@kwdef mutable struct GradDescent
    alpha::Float64 = 0.01
    error::Float64 = 0.0
    delta::Float64 = 0.0
    pred::Float64 = 0.0

    iter::Int16 = 1
    target = [1 1 0 1]
    weights= [0.1 0.2 -0.1;] 
    toes = [8.5 9.5 9.9 9.0]
    nfans = [1.2 1.3 0.5 1.0]
    wlrec = [0.65 0.8 0.8 0.9]
    inputs = [toes;wlrec;nfans]

    n_rows = size(weights,2) == 1 ? 1 : 1
    weighted_deltas = zeros(1,3)
    n_cols = size(inputs,2)

    predictions = zeros(n_rows, n_cols)
    
end

function step!(grad::GradDescent, col::Int64)
    println("Col $col")

    for iter in 1:10
        pred = grad.weights * grad.inputs[:,col]
        grad.pred = pred[1]
        grad.error = (grad.pred - grad.target[col]) ^ 2
        grad.delta = grad.pred - grad.target[col]
        grad.weighted_deltas[1,:] = round.(grad.delta * grad.inputs[:,col], digits=5)
        grad.weights -= grad.weighted_deltas .* grad.alpha
        grad.weights = round.(grad.weights, digits=5)
        println("Weights $(grad.weights)")
        println("Error: $(round(grad.error, digits=3)), Prediction: $(round(grad.pred, digits=3))") 
    end

    grad.predictions[col] = grad.pred
end

grad_descent = GradDescent()

for _col in 1:grad_descent.n_cols
    step!(grad_descent,_col)
end

