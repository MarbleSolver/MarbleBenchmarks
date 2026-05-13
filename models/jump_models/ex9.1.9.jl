# ex9.1.9.mod LLR-AY-NLP-12-11-5
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.10 in the Test Collection Book
# Test problem 9.1.9 in the web page
# Test Problem from visweswaran-etal 1996
# Taken from Bard 1983

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:5

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, x + y)

    # constraints
    @constraint(model, c2,       -x - 0.5*y + s[1] == -2)
    @constraint(model, c3,  -0.25*x +     y + s[2] ==  2)
    @constraint(model, c4,        x + 0.5*y + s[3] ==  8)
    @constraint(model, c5,        x -   2*y + s[4] ==  2)
    @constraint(model, c6, -y + s[5] == 0)

    @constraint(model, kt1, -0.5*l[1] + l[2] + 0.5*l[3] - 2*l[4] - l[5] == 1)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
