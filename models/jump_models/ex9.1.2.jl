# ex9.1.2.mod LLR-AY-NLP-10-9-4
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.3 in the Test Collection Book
# Test problem 9.1.2 in the web page
# Test problem from Liu and Hart 1994

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:4

    @variable(model, x >= 0)
    @variable(model, y, Bin)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, -x - 3*y)

    # constraints
    @constraint(model, c2, -x +   y + s[1] == 3)
    @constraint(model, c3,  x + 2*y + s[2] == 12)
    @constraint(model, c4, 4*x -   y + s[3] == 12)
    @constraint(model, c5,       -  y + s[4] == 0)

    @constraint(model, kt1, l[1] + 2*l[2] - l[3] - l[4] == -1)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
