# ex9.1.1.mod LLR-AY-NLP-13-12-5
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.2 in the Test Collection Book
# Test problem 9.1.1 in the web page
# Test problem from Clark and Westerberg 1990
#
# Is there a mistake in constraint kt2? l[2] appears twice!

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:5

    @variable(model, y1)
    @variable(model, y2)
    @variable(model, x >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, -x - 3*y1 + 2*y2)

    # constraints
    @constraint(model, c1, -2*x +   y1 + 4*y2 + s[1] == 16)
    @constraint(model, c2,  8*x + 3*y1 - 2*y2 + s[2] == 48)
    @constraint(model, c3, -2*x +   y1 - 3*y2 + s[3] == -12)
    @constraint(model, c4,       -  y1         + s[4] == 0)
    @constraint(model, c5,          y1         + s[5] == 4)

    @constraint(model, kt1, -1 + l[1] + 3*l[2] + l[3] - l[4] + l[5] == 0)
    @constraint(model, kt2,           4*l[2] - 2*l[2] - 3*l[3] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
