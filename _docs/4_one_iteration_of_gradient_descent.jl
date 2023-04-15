# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 59

using Plots

function first_iteration_of_gradient_descent()
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

function second_iteration_of_gradient_descent()
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

function third_iteration_of_gradient_descent()
    input = 8.5
    weight = 0.1
    target = 1.0
    alpha = 0.01
    
    iter_range = 1:10
    
    @gif for iter in iter_range
        pred = input * weight
        error = (pred - target) ^ 2
        weight -= (input * (pred - target)) * alpha
        
        plt = plot(
            [0, 10],
            [weight * 0,
            weight * 10],
            label="Prediction")
        plot!([0, 10], [0, 10], label="Target")
        plot!([input, input], [0, pred], linestyle=:dash, label="")
        title!("Iteration: $iter, Error: $(round(error, digits=5))")
        xlabel!("Input")
        ylabel!("Output")
    end
end

Base.@kwdef mutable struct GradDescent
    input::Float64 = 8.5
    weight::Float64 = 0.1
    target::Int128 = 10
    alpha::Float64 = 0.01
    pred::Float64 = 0.0
    error::Float64 = 0.0
    iter::Int128 = 1
    x::Float64 = 1
    y::Float64 = 1
end

function step!(l::GradDescent)
    l.pred = l.input * l.weight
    l.error = round((l.pred - l.target) ^ 2, digits=3)
    l.weight -= (l.input * (l.pred - l.target)) * l.alpha
    l.y+=1
    l.x += 1

    println("X: $(round(l.x, digits=2)) Y: $(l.y)")
end

grad_descent = GradDescent()

plt = plot(
    1,
    xlim = (0,10),
    ylim = (0,100),
    title = "Gradual Descent",
    legend = false,
    marker = 3,
)

@gif for iter in 1:10
    step!(grad_descent)
    push!(plt,grad_descent.x, grad_descent.y)

end every 10