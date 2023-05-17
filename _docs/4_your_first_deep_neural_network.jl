# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 125

using Random

Random.seed!(1)

function relu(x)
    return max.(x, 0)
end

function relu2deriv(output)
    return output .> 0
end

streetlights = [
    1 0 1;
    0 1 1;
    0 0 1;
    1 1 1]

walk_vs_stop = [1; 1; 0; 0]

alpha = 0.1
hidden_size = 4

weights_0_1 = 2 * rand(3, hidden_size) .- 1
weights_1_2 = 2 * rand(hidden_size, 1) .- 1

for iteration in 1:100
    layer_2_error = 0
    for i in 1:size(streetlights, 1)
        layer_0 = streetlights[i:i, :]
        layer_1 = relu(layer_0 * weights_0_1)
        layer_2 = layer_1 * weights_1_2
        layer_2_error += sum((layer_2 .- walk_vs_stop[i:i]) .^ 2)
        layer_2_delta = (layer_2 .- walk_vs_stop[i:i])
        layer_1_delta = layer_2_delta * weights_1_2' .* relu2deriv(layer_1)
        weights_1_2 .-= alpha * layer_1' * layer_2_delta
        weights_0_1 .-= alpha * layer_0' * layer_1_delta
    end
    if iteration % 10 == 0
        println("Iteration: $iteration - Error: $layer_2_error")
    end
end




    


    



