# ex9.2.3.mod LLR-AY-NLP-16-16-6
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.4 in the Test Collection Book
# Test problem 9.2.3 in the web page
# Test problem from Visweswaran etal 1996
# Originally from Shimizu Aiyoshi 81
# Ouadratic Outer and Inner Objective - Nonlinear

#********************************************
# This program locates the LOCAL minimum
#********************************************

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:6

    @variable(model, y1 >= -8)
    @variable(model, y2 >= -8)
    @variable(model, 1 <= x1 <= 50)
    @variable(model, 1 <= x2 <= 50)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, 2*x1 + 2*x2 - 3*y1 - 3*y2 - 60)

    # constraints
    @constraint(model, o1, x1 + x2 + y1 - 2*y2 <= 40)

    @constraint(model, c1, -x1 + 2*y1 + s[1] == -10)
    @constraint(model, c2, -x2 + 2*y2 + s[2] == -10)
    @constraint(model, c3,      -   y1 + s[3] == 10)
    @constraint(model, c4,          y1 + s[4] == 20)
    @constraint(model, c5,      -   y2 + s[5] == 10)
    @constraint(model, c6,          y2 + s[6] == 20)

    @constraint(model, kt1, 2*(y1 - x1 + 20) + 2*l[1] - l[3] + l[4] == 0)
    @constraint(model, kt2, 2*(y2 - x2 + 20) + 2*l[2] - l[5] + l[6] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
