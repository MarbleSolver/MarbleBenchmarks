
# MPEC example taken from Gauvin and Savard, "Working Paper G9037",
# GERAD, Ecole Polytechnique de Montreal (1992) (first version).
#
#    Let Y(x) := { y | 0 <= y <= 20 - x }.
#    min  f(x,y) := x**2 + (y-10)**2
#    s.t.      0 <= x <= 15
#              y solves MCP((F(y) := 4(x + 2y - 30)), Y(x))
#
# We have modified the problem by adding a dual variable and
# incorporating the constraint y <= 20 - x into the MCP function F.
#
# From a GAMS model by S.P. Dirkse & M.C. Ferris (MPECLIB),
# (see http://www.gams.com/mpec/).
#
# AMPL coding Sven Leyffer, University of Dundee, Jan. 2000

using JuMP

function build_model()
    model = Model()

    @variable(model, 0 <= x <= 15)
    @variable(model, y >= 0)
    @variable(model, u >= 0)

    @objective(model, Min, x^2 + (y-10)^2)

    # ... F(x,y) from original problem
    @variable(model, Fy_s >= 0)
    @constraint(model, Fy_def, Fy_s == 4 * (x + 2*y - 30) + u)
    @constraint(model, Fy, [Fy_s, y] in SOS1())

    @variable(model, Fu_s >= 0)
    @constraint(model, Fu_def, Fu_s == 20 - x - y)
    @constraint(model, Fu, [Fu_s, u] in SOS1())

    return model
end

model = build_model()
