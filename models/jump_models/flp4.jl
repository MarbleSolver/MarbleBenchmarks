# flp4.mod	QLR-AN-LCP-v-v-v
#
# Problem 4 from Fukushima, M. Luo, Z.-Q.Pang, J.-S.,
# "A globally convergent Sequential Quadratic Programming
# Algorithm for Mathematical Programs with Linear Complementarity
# Constraints", Computational Optimization and Applications, 10(1),
# pp. 5-34, 1998.
#
# This is a QPEC with random data of the form
#
#	minimize    0.5*x^T x + e^T y
#
#	subject to  A x <= b
# 		    0 <= y  _|_  N x + M y + q >= 0

using JuMP

function build_model(m::Int, n::Int, p::Int, A, b, N_mat, M_mat, q_vec)
    model = Model()

    @variable(model, x[1:n])
    @variable(model, y[1:m] >= 0)

    @objective(model, Min, 0.5 * sum(x[j]^2 for j in 1:n) + sum(y[i] for i in 1:m))

    @constraint(model, lincs[k in 1:p], sum(A[k,j]*x[j] for j in 1:n) <= b[k])

    @variable(model, compl_s[1:m] >= 0)
    @constraint(model, compl_def[i in 1:m],
        compl_s[i] == sum(N_mat[i,j]*x[j] for j in 1:n) + sum(M_mat[i,l]*y[l] for l in 1:m) + q_vec[i])
    @constraint(model, compl[i in 1:m], [y[i], compl_s[i]] in SOS1())

    return model
end
