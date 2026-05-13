# ex9.2.1.mod QLR-AY-NLP-10-9-4
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.2 in the Test Collection Book
# Test problem 9.2.1 in the web page
# Test problem from Shimizu etal 1997

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:4

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, (x - 5)*(x - 5) + (2*y + 1)*(2*y + 1))

    # constraints
    @constraint(model, c1, -3*x +     y + s[1] == -3)
    @constraint(model, c2,    x - 0.5*y + s[2] ==  4)
    @constraint(model, c3,    x +     y + s[3] ==  7)
    @constraint(model, c4,          - y + s[4] ==  0)

    @constraint(model, kt1, 2*(y - 1) - 1.5*x + l[1] - 0.5*l[2] + l[3] - l[4] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
