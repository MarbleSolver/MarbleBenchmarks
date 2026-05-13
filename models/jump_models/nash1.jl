# nash1.mod	QQR2-MN-8-5
# Original AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 9

# Number of variables:  6 + 2 slacks
# Number of constraints: 4

using JuMP

function build_model()
    model = Model()

    @variable(model, 0 <= x[1:2] <= 10)
    @variable(model, y[1:2])
    @variable(model, l[1:2] >= 0)

    @objective(model, Min, ((x[1] - y[1])^2 + (x[2] - y[2])^2) / 2)

    @constraint(model, F1, 0 == -34 + 2*y[1] + (8/3)*y[2] - (-l[1]))
    @constraint(model, F2, 0 == -24.25 + 1.25*y[1] + 2*y[2] - (-l[2]))

    @variable(model, g1_s >= 0)
    @constraint(model, g1_def, g1_s == -x[2] - y[1] + 15)
    @constraint(model, g1, [g1_s, l[1]] in SOS1())

    @variable(model, g2_s >= 0)
    @constraint(model, g2_def, g2_s == -x[1] - y[2] + 15)
    @constraint(model, g2, [g2_s, l[2]] in SOS1())

    return model
end

model = build_model()
