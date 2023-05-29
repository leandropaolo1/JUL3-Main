# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 125

Base.@kwdef mutable struct StreetLight
    alpha::Float64 = 0.1
    iter::Int16 = 1

    targets = [0, 1, 0, 1, 1, 0]
    weights = [0.5, 0.48 , -0.7]

    inputs = [
        1 0 1;
        0 1 1;
        0 0 1;
        1 1 1;
        0 1 1;
        1 0 1;]
    
        n_rows = size(inputs, 1)
        n_cols = size(inputs, 2)
        preds = zeros(n_rows, 1)
        errors = zeros(n_rows, 1)
        deltas = zeros(n_rows, 1)
        weighted = zeros(3, 1)
end

function step!(light::StreetLight, row)
    for iter in 1:40
        light.preds[row] = sum(light.weights .* light.inputs[row,:])
        light.deltas[row] = light.preds[row] - light.targets[row]
        light.weighted = light.inputs[row,:] .* light.deltas[row] 
        light.weights -= light.alpha .* light.weighted

    end
    println("Target: $(light.targets[row])")
    println("Prediction: $(round.(light.preds[row], digits=3))")
end

_light = StreetLight()

for iter in 1:_light.n_rows
    step!(_light, iter)
end

    


    



#= Iteration 1

Base.@kwdef mutable struct StreetLight
    alpha::Float64 = 0.1
    iter::Int16 = 1

    targets = [0, 1, 0, 1, 1, 0]
    weights = [0.5, 0.48 , -0.7]

    inputs = [
        1 0 1;
        0 1 1;
        0 0 1;
        1 1 1;
        0 1 1;
        1 0 1;]
    
        n_rows = size(inputs, 1)
        n_cols = size(inputs, 2)
        preds = zeros(n_rows, 1)
        errors = zeros(n_rows, 1)
        deltas = zeros(n_rows, 1)
        weighted = zeros(3, 1)
end

function step!(light::StreetLight, row)
    for iter in 1:50
        light.preds[row] = sum(light.weights .* light.inputs[row,:])
        light.deltas[row] = light.preds[row] - light.targets[row]
        light.weighted = light.inputs[row,:] .* light.deltas[row] 
        light.weights -= light.alpha .* light.weighted

    end
    println("Target: $(light.targets[row])")
    println("Prediction: $(round.(light.preds[row], digits=3))")
end

_light = StreetLight()

for iter in 1:_light.n_rows
    step!(_light, iter)
end

=#

#= Iteration 2
Base.@kwdef mutable struct StreetLight
    alpha::Float64 = 0.1
    iter::Int16 = 1

    targets = [0 1 0 1 1 0;]
    weights = [0.5 0.48 -0.7;]

    inputs = [
        1 0 1;
        0 1 1;
        0 0 1;
        1 1 1;
        0 1 1;
        1 0 1;]

    
    pred = zeros(size(inputs))
end


st = StreetLight()

for iteration in 1:40
    st.pred = st.weights .* st.inputs
    
=#