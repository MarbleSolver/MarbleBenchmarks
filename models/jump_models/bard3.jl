# bard3.mod	QQR2-MN-8-6
# Original AMPL coding by Sven Leyffer

# An MPEC from J.F. Bard, Convex two-level optimization,
# Mathematical Programming 40(1), 15-27, 1988.

# Number of variables:   4 + 2 multipliers
# Number of constraints: 5

using JuMP

function build_model()
    model = Model()

    N = 1:2

    @variable(model, x[N] >= 0)
    @variable(model, y[N] >= 0)

    # ... slack variables & multipliers
    @variable(model, l[N] >= 0)

    @objective(model, Min, - x[1]^2 - 3*x[2] - 4*y[1] + y[2]^2)

    @constraint(model, nlncs, x[1]^2 + 2*x[2] <= 4)

    @constraint(model, KKT1, 2*y[1] + l[1]*2 - l[2]*3 == 0)
    @constraint(model, KKT2, -5 - l[1] + l[2]*4 == 0)

    @variable(model, lin_1_s >= 0)
    @constraint(model, lin_1_def, lin_1_s == x[1]^2 - 2*x[1] + x[2]^2 - 2*y[1] + y[2] + 3)
    @constraint(model, lin_1, [lin_1_s, l[1]] in SOS1())

    @variable(model, lin_2_s >= 0)
    @constraint(model, lin_2_def, lin_2_s == x[2] + 3*y[1] - 4*y[2] - 4)
    @constraint(model, lin_2, [lin_2_s, l[2]] in SOS1())

    return model
end

model = build_model()
