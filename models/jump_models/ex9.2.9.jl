# ex9.2.9.mod LLR-AY-NLP-9-8-3
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.10 in the Test Collection Book
# Test problem 9.2.9 in the web page
# Test problem from Bard 91

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:3

    @variable(model, 2 <= x <= 4)
    @variable(model, y1 >= 0)
    @variable(model, y2 >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, x + y2)

    # constraints
    @constraint(model, c1,   x - y1 - y2 + s[1] == -4)
    @constraint(model, c2,     - y1       + s[2] == 0)
    @constraint(model, c3,     - y2       + s[3] == 0)

    @constraint(model, kt1, -l[1] - l[2] == -2)
    @constraint(model, kt2, -l[1] - l[3] == -x)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
