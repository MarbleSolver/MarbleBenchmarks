# bard1.mod	QQR2-MN-8-5
# Original AMPL coding by Sven Leyffer

# An MPEC from J.F. Bard, Convex two-level optimization,
# Mathematical Programming 40(1), 15-27, 1988.

# Number of variables:   2 + 3 slack + 3 multipliers
# Number of constraints: 4

using JuMP

model = Model()

@variable(model, x >= 0)
@variable(model, y >= 0)

# ... multipliers
@variable(model, l[1:3] >= 0)

@objective(model, Min, (x - 5)^2 + (2*y + 1)^2)

@variable(model, s[1:3] >= 0)

@constraint(model, KKT, 2*(y - 1) - 1.5*x + l[1] - 0.5*l[2] + l[3] == 0)

@constraint(model, lin_1_expr, s[1] == 3*x - y - 3)
@constraint(model, lin_2_expr, s[2] == -x + 0.5*y + 4)
@constraint(model, lin_3_expr, s[3] == -x - y + 7)

@constraint(model, lin_1, [s[1], l[1]] in SOS1())
@constraint(model, lin_2, [s[2], l[2]] in SOS1())
@constraint(model, lin_3, [s[3], l[3]] in SOS1())
