# ex9.1.3.mod LLR-AY-NLP-23-21-6
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.3 in the Test Collection Book
# Test problem 9.1.3 in the web page
# Test problem from Candler-Townsley 82
#
# Removed MI-model of complementarity.

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:6
    J = 1:3

    @variable(model, y[I] >= 0)
    @variable(model, mu[J])
    @variable(model, x[J] >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, 4*y[1] - 40*y[2] - 4*y[3] - 8*x[1] - 4*x[2])

    # constraints
    @constraint(model, c2,  -y[1] +   y[2] +     y[3] + y[4]                    == 1)
    @constraint(model, c3,  -y[1] + 2*y[2] - 0.5*y[3]      + y[5]      + 2*x[1] == 1)
    @constraint(model, c4, 2*y[1] -   y[2] - 0.5*y[3]           + y[6] + 2*x[2] == 1)
    @constraint(model, c5,  -y[1] + s[1] == 0)
    @constraint(model, c6,  -y[2] + s[2] == 0)
    @constraint(model, c7,  -y[3] + s[3] == 0)
    @constraint(model, c8,  -y[4] + s[4] == 0)
    @constraint(model, c9,  -y[5] + s[5] == 0)
    @constraint(model, c10, -y[6] + s[6] == 0)

    @constraint(model, kt1,  1 - mu[1] -     mu[2] +   2*mu[3] - l[1] == 0)
    @constraint(model, kt2,  1 + mu[1] +   2*mu[2] -     mu[3] - l[2] == 0)
    @constraint(model, kt3,  2 + mu[1] - 0.5*mu[2] - 0.5*mu[3] - l[3] == 0)
    @constraint(model, kt4,  mu[1] - l[4] == 0)
    @constraint(model, kt5,  mu[2] - l[5] == 0)
    @constraint(model, kt6,  mu[3] - l[6] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
