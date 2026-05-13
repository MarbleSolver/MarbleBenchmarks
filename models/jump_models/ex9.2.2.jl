# ex9.2.2.mod QLR-AY-NLP-10-11-4
# AMPL coding by Sven Leyffer, Apr. 2001,

# From Nonconvex Optimization and its Applications, Volume 33
# Kluwer Academic Publishers, Dordrecht, Hardbound, ISBN 0-7923-5801-5
# (see also titan.princeton.edu/TestProblems/)
#
# Test problem 9.3.3 in the Test Collection Book
# Test problem 9.2.2 in the web page
# Test problem from Visweswaran etal 1996
# Originally from Shimizu Aiyoshi 81

using JuMP

function build_model()
    model = Model()

    # variables
    I = 1:4

    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, s[I] >= 0)
    @variable(model, l[I] >= 0)

    # objective
    @objective(model, Min, x*x + (y - 10)*(y - 10))

    # constraints
    @constraint(model, o1,   x     <= 15)
    @constraint(model, o2, - x + y <= 0)
    @constraint(model, o3, - x     <= 0)

    @constraint(model, c1, x + y + s[1] == 20)
    @constraint(model, c2,   - y + s[2] == 0)
    @constraint(model, c3,     y + s[3] == 20)

    @constraint(model, kt1, 2*(x + 2*y - 30) + l[1] - l[2] + l[3] == 0)

    @constraint(model, compl[i in I], [l[i], s[i]] in SOS1())

    return model
end

model = build_model()
