# ex9.2.5.mod QLR-AY-NLP-8-7-3
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.6 in the Test Collection Book
# Test problem 9.2.5 in the web page
# Test problem from Clark and Westerberg 1990a

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:3

    @variable(model, y)
    @variable(model, 0 <= x <= 8)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, (x - 3)*(x - 3) + (y - 2)*(y - 2))

    # constraints
    @constraint(model, c1, -2*x +   y + s[1] == 1)
    @constraint(model, c2,    x - 2*y + s[2] == 2)
    @constraint(model, c3,    x + 2*y + s[3] == 14)

    @constraint(model, kt1, 2*(y - 5) + l[1] - 2*l[2] + 2*l[3] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
