

function n_mat(n, p)
    mat = fill(p, 1, n)
    output = join(mat, " ")
    println(output)
end

num = [5, 2, 3, 2, 7, 5, 2, 5, 3, 5, 2, 5, 3, 5, 3, 3]

ii = 0
for i in 1:19
    

    n_mat(num[i], ii)
    ii += 1
end

num = [2,4,3,7,6,3,2,6,4,1,1,7,2,2,2,2,2,3,1]