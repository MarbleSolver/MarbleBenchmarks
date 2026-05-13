# bard2.mod	QQR2-MN-
# Original AMPL coding by Sven Leyffer

# An MPEC from J.F. Bard, Convex two-level optimization,
# Mathematical Programming 40(1), 15-27, 1988.
# From Aiyoshi & Shimizu, IEEE Trans Syst Man & Cyb SMC-11 (1981),
# 444-449.

# Corrected index error in constraint lin_12 (S Leyffer)

# Number of variables:   8 + 4 slack + 4 multipliers
# Number of constraints: 9
# Nonlinear complementarity constraints

using JuMP

function build_model(;
    u_x=Dict((1,1)=>10,(1,2)=>5,(2,1)=>15,(2,2)=>20),
    u_y=Dict((1,1)=>20,(1,2)=>20,(2,1)=>40,(2,2)=>40))

    model = Model()

    N = 1:2

    @variable(model, 0 <= x[i in N, j in N] <= u_x[i,j])
    @variable(model, 0 <= y[i in N, j in N] <= u_y[i,j])

    # ... multipliers
    @variable(model, l[N, N] >= 0)

    @objective(model, Max,   (200 - y[1,1] - y[2,1])*(y[1,1] + y[2,1])
                           + (160 - y[1,2] - y[2,2])*(y[1,2] + y[2,2]))

    @constraint(model, lincs, x[1,1] + x[1,2] + x[2,1] + x[2,2] <= 40)

    @constraint(model, KKT1_1, 2*(y[1,1] - 4)  + l[1,1]*0.4 + l[1,2]*0.6 == 0)
    @constraint(model, KKT1_2, 2*(y[1,2] - 13) + l[1,1]*0.7 + l[1,2]*0.3 == 0)

    @variable(model, lin_11_s >= 0)
    @constraint(model, lin_11_def, lin_11_s == x[1,1] - 0.4*y[1,1] - 0.7*y[1,2])
    @constraint(model, lin_11, [lin_11_s, l[1,1]] in SOS1())

    @variable(model, lin_12_s >= 0)
    @constraint(model, lin_12_def, lin_12_s == x[1,2] - 0.6*y[1,1] - 0.3*y[1,2])
    @constraint(model, lin_12, [lin_12_s, l[1,2]] in SOS1())

    @constraint(model, KKT2_1, 2*(y[2,1] - 35)  + l[2,1]*0.4 + l[2,2]*0.6 == 0)
    @constraint(model, KKT2_2, 2*(y[2,2] - 2) + l[2,1]*0.7 + l[2,2]*0.3 == 0)

    @variable(model, lin_21_s >= 0)
    @constraint(model, lin_21_def, lin_21_s == x[2,1] - 0.4*y[2,1] - 0.7*y[2,2])
    @constraint(model, lin_21, [lin_21_s, l[2,1]] in SOS1())

    @variable(model, lin_22_s >= 0)
    @constraint(model, lin_22_def, lin_22_s == x[2,2] - 0.6*y[2,1] - 0.3*y[2,2])
    @constraint(model, lin_22, [lin_22_s, l[2,2]] in SOS1())

    return model
end

model = build_model()
