# ex9.2.6.mod QLR-AY-NLP-16-12-6
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.7 in the Test Collection Book
# Test problem 9.2.6 in the web page
# Test problem from Falk and Liu 1995
# Original example from A.D. De Silva's dissertation 78
# Ouadratic Outer and Inner Objective - Nonlinear

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:6

    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)
    @variable(model, y1 >= 0)
    @variable(model, y2 >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, x1*x1 - 2*x1 + x2*x2 - 2*x2 + y1*y1 + y2*y2)

    # constraints
    @constraint(model, c1, 0.5 - y1 + s[1] == 0)
    @constraint(model, c2, 0.5 - y2 + s[2] == 0)
    @constraint(model, c3, y1 - 1.5 + s[3] == 0)
    @constraint(model, c4, y2 - 1.5 + s[4] == 0)

    @constraint(model, kt1, 2*(y1 - x1) - l[1] + l[3] == 0)
    @constraint(model, kt2, 2*(y2 - x2) - l[2] + l[4] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
