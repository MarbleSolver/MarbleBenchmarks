# ex9.1.6.mod LLR-AY-NLP-14-13-6
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.7 in the Test Collection Book
# Test problem 9.1.6 in the web page
# Test Problem from Anandalingam and White 1990
# They took from Bialas and Karwan 1982

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:6

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, -x - 3*y)

    # constraints
    @constraint(model, c1, - x - 2*y + s[1] == -10)
    @constraint(model, c2,   x - 2*y + s[2] == 6)
    @constraint(model, c3, 2*x -   y + s[3] == 21)
    @constraint(model, c4,   x + 2*y + s[4] == 38)
    @constraint(model, c5,  -x + 2*y + s[5] == 18)
    @constraint(model, c6,        - y + s[6] == 0)

    @constraint(model, kt1, 3 - 2*l[1] - 2*l[2] - l[3] + 2*l[4] + 2*l[5] - l[6] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
