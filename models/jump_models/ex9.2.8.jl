# ex9.2.8.mod QLR-AY-NLP-6-5-2
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.9 in the Test Collection Book
# Test problem 9.2.8 in the web page
# Test problem from Yezza 96
# Bilinear Inner Objective

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:2

    @variable(model, 0 <= x <= 1)
    @variable(model, y >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, -4*x*y + 3*y + 2*x + 1)

    # constraints
    @constraint(model, c1, -y + s[1] == 0)
    @constraint(model, c2,  y + s[2] == 1)

    @constraint(model, kt1, -(1 - 4*x) - l[1] + l[2] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
