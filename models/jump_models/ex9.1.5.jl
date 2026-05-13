# ex9.1.5.mod LLR-AY-NLP-13-12-5
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.2.5 in the Test Collection Book
# Test problem 9.1.5 in the web page
# Test problem from Bard 91
# Note there is a typo in the Book corrected here

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:5

    @variable(model, x >= 0)
    @variable(model, y1 >= 0)
    @variable(model, y2 >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, -x + 10*y1 - y2)

    # constraints
    @constraint(model, c1,   x + y1 + s[1] == 1)
    @constraint(model, c2,   x + y2 + s[2] == 1)
    @constraint(model, c3,  y1 + y2 + s[3] == 1)
    @constraint(model, c4,  -y1 + s[4] == 0)
    @constraint(model, c5,  -y2 + s[5] == 0)

    @constraint(model, kt1, l[1] + l[3] - l[4] == 1)
    @constraint(model, kt2, l[2] + l[3] - l[5] == 1)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
