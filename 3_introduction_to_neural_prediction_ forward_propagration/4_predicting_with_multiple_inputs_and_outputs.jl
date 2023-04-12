"""
    predicting_with_multiple_inputs_and_outputs_2()


# Grokking Deep Learning (Andrew W. Trask) (z-lib.org)
# PDF pg. 38
Return the predictions of a neural network with multiple inputs and outputs.

# Inputs
- `toes::Array{Float64,1}`: The number of toes a player has.
- `wlrec::Array{Float64,1}`: The current win/loss record.
- `nfans::Array{Float64,1}`: The fan count in millions.

# Outputs
- `prediction::Array{Float64,2}`: The predictions from the neural network.

# Example
```julia
prediction = predicting_with_multiple_inputs_and_outputs_2()

"""
function predicting_with_multiple_inputs_and_outputs_2()
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes; wlrec; nfans]

    weights = [
        0.1 0.1 -0.3
        0.1 0.2 0.0
        0.0 1.3 0.1]

    n_rows = size(inputs, 1)
    n_cols = size(inputs, 2)

    prediction = zeros(n_rows, n_cols)

    for iter in 1:n_cols
        col = inputs[:, iter]
        prediction[:, iter] = round.(weights * col, digits=3)

    end

    return prediction

end


begin
    predicting_with_multiple_inputs_and_outputs_2()
end

