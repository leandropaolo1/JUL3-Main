# how do you take in oen value or inputs and output multiple prediction? Lets find out. This of course is a really simple neural network
# p.36 pdf.57


begin
    results = []
    toes = [8.5 9.5 9.9 9.0]
    wlrec = [0.65 0.8 0.8 0.9]
    nfans = [1.2 1.3 0.5 1.0]
    inputs = [toes;wlrec;nfans]
    weights = [0.3, 0.2, 0.9] 

    hurt_prediction = []
    win_prediction = []
    sad_prediction = []

    for item in eachindex(wlrec)
        array_product = wlrec[item] * weights
        append!(hurt_prediction, round(array_product[1], digits=3))
        append!(win_prediction, round(array_product[2], digits=3))
        append!(sad_prediction, round(array_product[3], digits=3))

    end
    println(hurt_prediction)
    println(win_prediction)
    println(sad_prediction)
end

