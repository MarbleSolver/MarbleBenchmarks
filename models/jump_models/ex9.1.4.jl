# ex9.1.4.mod  LLR-AY-10-9-4
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.5 in the Test Collection Book
# Test problem 9.1.4 in the web page
# Test problem from Clark and Westerberg 1988

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
    @objective(model, Min, x - 4*y)

    # constraints
    @constraint(model, c2, -2*x +   y + s[1] == 0)
    @constraint(model, c3,  2*x + 5*y + s[2] == 108)
    @constraint(model, c4,  2*x - 3*y + s[3] == -4)
    @constraint(model, c5,        - y + s[4] == 0)

    @constraint(model, kt1, l[1] + 5*l[2] - 3*l[3] - l[4] == -1)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
