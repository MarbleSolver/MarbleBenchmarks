# sl1.mod	QQR2-MN-11-6
# Original AMPL coding by Sven Leyffer

# A QPEC obtained by varying the rhs of HS21 (a QP)
# from an idea communicated by S. Scholtes

# Number of variables:   5 + 3 multipliers
# Number of constraints: 5

using JuMP

function build_model(; zl=[10.0, 0.01, 0.0], zu=[1e10, 10.0, 1.0])
    model = Model()

    @variable(model, x[1:2])
    @variable(model, z[i in 1:3], lower_bound=zl[i], upper_bound=zu[i])

    # ... slack variables & multipliers
    @variable(model, l[1:3] >= 0)

    @objective(model, Min, (x[1] - 2)^2 + x[2]^2)

    @constraint(model, KKT1, 0.02*x[1] - 10*l[1] - l[2] == 0)
    @constraint(model, KKT2, 2*x[2] - l[1] - l[3] == 0)

    @variable(model, lin_1_s >= 0)
    @constraint(model, lin_1_def, lin_1_s == 10*x[1] + x[2] - (10 + z[1]))
    @constraint(model, lin_1, [lin_1_s, l[1]] in SOS1())

    @variable(model, lin_2_s >= 0)
    @constraint(model, lin_2_def, lin_2_s == x[1] - (2 + z[2]))
    @constraint(model, lin_2, [lin_2_s, l[2]] in SOS1())

    @variable(model, lin_3_s >= 0)
    @constraint(model, lin_3_def, lin_3_s == x[2] - 50*z[3])
    @constraint(model, lin_3, [lin_3_s, l[3]] in SOS1())

    return model
end

model = build_model()
