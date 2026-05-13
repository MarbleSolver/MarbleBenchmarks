# ex9.1.8.mod LLR-AY-NLP-14-12-5
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.8 in the Test Collection Book
# Test problem 9.1.8 in the web page
# Test problem from Bard and Falk 82
# Note the correct results are given here

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:5

    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)
    @variable(model, y1 >= 0)
    @variable(model, y2 >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, -2*x1 + x2 + 0.5*y1)

    # constraints
    @constraint(model, c0, x1 + x2 <= 2)

    @constraint(model, c1, -2*x1 +   y1 - y2 + s[1] == -2.5)
    @constraint(model, c2,    x1 - 3*x2 + y2 + s[2] == 2)
    @constraint(model, c3,  -y1 + s[3] == 0)
    @constraint(model, c4,  -y2 + s[4] == 0)

    @constraint(model, kt1, l[1] - l[3] == 4)
    @constraint(model, kt2, l[1] + l[2] - l[4] == -1)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
