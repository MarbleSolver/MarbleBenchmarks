# liswet1-inv.mod  QLR-AN-KKT-v-v-v
# MPEC AMPL by S. Leyffer, University of Dundee, May 2002.
#
# A QPEC from an idea by Stefan Scholtes:
# Find minimum l_2 norm distance to original QP soln
# for perturbed right-hand-side
#
# Original NLP-AMPL Model by Hande Y. Benson
#
# Copyright (C) 2001 Princeton University
# All Rights Reserved
#
# ... from ...
#
#   Source:
#   W. Li and J. Swetits,
#   "A Newton method for convex regression, data smoothing and
#   quadratic programming with bounded constraints",
#   SIAM J. Optimization 3 (3) pp 466-488, 1993.
#
#   SIF input: Nick Gould, August 1994.

using JuMP

function build_model(N::Int, x_star::Vector{Float64})
    K = 2
    B = [factorial(i) for i in 0:K]
    C = [(-1)^i * B[K+1] / (B[i+1] * B[K-i+1]) for i in 0:K]
    T = [(i-1)/(N+K-1) for i in 1:(N+K)]

    model = Model()

    @variable(model, z[1:N] >= 0)
    @variable(model, x[1:(N+K)])
    @variable(model, l[1:N] >= 0)

    @objective(model, Min, sum((x[i] - x_star[i])^2 for i in 1:(N+K)))

    @constraint(model, KKT[i in 1:(N+K)],
        -(sqrt(T[i]) + 0.1*sin(i)) + x[i]
        - sum(C[j+K-i+1]*l[j] for j in max(i-K,1):min(i,N)) == 0)

    @constraint(model, controls, sum(z[j] for j in 1:N) >= 0.2)

    @variable(model, compl_s[1:N] >= 0)
    @constraint(model, compl_def[j in 1:N],
        compl_s[j] == sum(C[i+1]*x[j+K-i] for i in 0:K) - z[j])
    @constraint(model, compl[j in 1:N], [l[j], compl_s[j]] in SOS1())

    return model
end
