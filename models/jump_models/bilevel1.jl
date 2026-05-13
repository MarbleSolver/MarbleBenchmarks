# bilevel1.mod	QQR2-MN-16-10
# Original AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 7

# Number of variables:   10
# Number of constraints: 9

using JuMP

function build_model()
    model = Model()

    @variable(model, 0 <= x[1:2] <= 50)
    @variable(model, y[1:2])
    @variable(model, l[1:6] >= 0)

    @objective(model, Min, 2*x[1] + 2*x[2] - 3*y[1] - 3*y[2] - 60)

    @constraint(model, c1, x[1] + x[2] + y[1] - 2*y[2] - 40 <= 0)
    @constraint(model, F1, 0 == 2*y[1] - 2*x[1] + 40 - (l[1] - l[2] - 2*l[5]))
    @constraint(model, F2, 0 == 2*y[2] - 2*x[2] + 40 - (l[3] - l[4] - 2*l[6]))

    @variable(model, g1_s >= 0)
    @constraint(model, g1_def, g1_s == y[1] + 10)
    @constraint(model, g1, [g1_s, l[1]] in SOS1())

    @variable(model, g2_s >= 0)
    @constraint(model, g2_def, g2_s == -y[1] + 20)
    @constraint(model, g2, [g2_s, l[2]] in SOS1())

    @variable(model, g3_s >= 0)
    @constraint(model, g3_def, g3_s == y[2] + 10)
    @constraint(model, g3, [g3_s, l[3]] in SOS1())

    @variable(model, g4_s >= 0)
    @constraint(model, g4_def, g4_s == -y[2] + 20)
    @constraint(model, g4, [g4_s, l[4]] in SOS1())

    @variable(model, g5_s >= 0)
    @constraint(model, g5_def, g5_s == x[1] - 2*y[1] - 10)
    @constraint(model, g5, [g5_s, l[5]] in SOS1())

    @variable(model, g6_s >= 0)
    @constraint(model, g6_def, g6_s == x[2] - 2*y[2] - 10)
    @constraint(model, g6, [g6_s, l[6]] in SOS1())

    return model
end

model = build_model()
