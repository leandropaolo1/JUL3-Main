Base.@kwdef mutable struct StreetLight
    alpha::Float64 = 0.01
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
        pred = zeros(n_rows, 1)
        errors = zeros(n_rows, n_cols)
        delta = zeros(n_rows, n_cols)
        weighted = zeros(n_rows, n_cols)
end

function step!(light::StreetLight, row)
    light.pred[row,:] = 
end