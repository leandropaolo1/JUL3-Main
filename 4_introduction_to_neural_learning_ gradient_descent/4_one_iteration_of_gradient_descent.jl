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
        
        plt = plot([0, 10], [weight * 0, weight * 10], label="Prediction")
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
    target::Float64 = 1.0
    alpha::Float64 = 0.01
    pred::Float64 = 0.0
    error::Float64 = 0.0
end

function step!(l::GradDescent)
    l.pred = l.input * l.weight
    l.error = (l.pred - l.target) ^ 2
    l.weight -= (l.input * (l.pred - l.target)) * l.alpha
end

begin
    grad_descent = GradDescent()

    plt = plot3d(
        1,
        xlim = (-1.0, 1.0),
        ylim = (-1.0, 1.0),
        title = "Gradual Descent",
        legend = false,
        marker = 2,
    )

    @gif for i=1:10
        step!(grad_descent)
        push!(plt, grad_descent.target, grad_descent.error)
    end every 10
end