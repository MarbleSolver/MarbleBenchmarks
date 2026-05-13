# ex9.2.4.mod QLR-AY-NLP-8-7-2
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.5 in the Test Collection Book
# Test problem 9.2.4 in the web page
# Test problem from Yezza 1996

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:2

    @variable(model, l1)
    @variable(model, x >= 0)
    @variable(model, y1 >= 0)
    @variable(model, y2 >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, 0.5*(y1 - 2)*(y1 - 2) + 0.5*(y2 - 2)*(y2 - 2))

    # constraints
    @constraint(model, c1,   y1 + y2 == x)
    @constraint(model, c2, -y1       + s[1] == 0)
    @constraint(model, c3,      - y2 + s[2] == 0)

    @constraint(model, kt1, y1 + l1 - l[1] == 0)
    @constraint(model, kt2,  1 + l1 - l[2] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
