# so what is the simplest neural network available? Lets find out!

begin
    function simple_neural_network(input, weight)
        prediction = input * weight
        return prediction
    end
    input = 8.5
    weight = 0.1
    simple_neural_network(input,weight)
end