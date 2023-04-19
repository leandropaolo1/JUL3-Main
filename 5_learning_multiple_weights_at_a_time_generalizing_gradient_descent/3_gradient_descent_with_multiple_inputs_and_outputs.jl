Base.@kwdef mutable struct GradDescent
    alpha::Float64 = 0.01
    iter::Int16 = 1

    targets = [
        0.1 0.0 0.0 0.1;
        1.0 1.0 0.0 1.0;
        0.1 0.0 0.1 0.2]

    weights = [
        0.3 1.1 -0.3;
        0.1 0.2 0.0;
        0.0 1.3 0.1;] 

    inputs = [
        8.5 9.5 9.9 9.0;
        0.65 0.8 0.8 0.9;
        1.2 1.3 0.5 1.0]
    
    n_cols = size(inputs, 2)
    n_rows = size(inputs, 1)
    pred = zeros(n_rows, n_cols)
    errors = zeros(n_rows, n_cols)
    delta = zeros(n_rows, n_cols)
    weighted = zeros(n_rows, n_cols)
    prediction = zeros(n_rows, n_cols)
    

end


function step!(grad::GradDescent, col::Int8)
    for iter in 1:10
        grad.pred[:,col] = round.(grad.weights * grad.inputs[:,col], digits=5)
        grad.errors[:,col] = round.(grad.pred[:,col] .- grad.targets[:,col] .^ 2, digits=5)
        grad.delta[:,col] = grad.pred[:,col] .- grad.targets[:,col]
        grad.weighted[:,col] = round.(grad.delta[:,col] .* grad.inputs[:,col], digits=5)



end