# bilevel2.mod	QQR2-MN-32-18
# Original AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 10

# Number of variables:   20 + 12 slacks
# Number of constraints: 17
# Nonlinear complementarity constraint

using JuMP

function build_model(; ubx=[10, 5, 15, 20])
    model = Model()

    I = 1:4

    @variable(model, 0 <= x[i in I] <= ubx[i])
    @variable(model, y[I])
    @variable(model, l[1:12] >= 0)

    @objective(model, Min, - (200 - y[1] - y[3])*(y[1] + y[3]) - (160 - y[2] - y[4])*(y[2] + y[4]))

    @constraint(model, l1, x[1] + x[2] + x[3] + x[4] <= 40)
    @constraint(model, F1, 0 == y[1] - 4  - ( - 0.4*l[1] - 0.6*l[2] + l[3] - l[4]))
    @constraint(model, F2, 0 == y[2] - 13 - ( - 0.7*l[1] - 0.3*l[2] + l[5] - l[6]))
    @constraint(model, F3, 0 == y[3] - 35 - ( - 0.4*l[7] - 0.6*l[8] + l[9] - l[10]))
    @constraint(model, F4, 0 == y[4] - 2  - ( - 0.7*l[7] - 0.3*l[8] + l[11] - l[12]))

    @variable(model, g1_s >= 0)
    @constraint(model, g1_def, g1_s == x[1] - 0.4*y[1] - 0.7*y[2])
    @constraint(model, g1, [g1_s, l[1]] in SOS1())

    @variable(model, g2_s >= 0)
    @constraint(model, g2_def, g2_s == x[2] - 0.6*y[1] - 0.3*y[2])
    @constraint(model, g2, [g2_s, l[2]] in SOS1())

    @variable(model, g3_s >= 0)
    @constraint(model, g3_def, g3_s == y[1])
    @constraint(model, g3, [g3_s, l[3]] in SOS1())

    @variable(model, g4_s >= 0)
    @constraint(model, g4_def, g4_s == - y[1] + 20)
    @constraint(model, g4, [g4_s, l[4]] in SOS1())

    @variable(model, g5_s >= 0)
    @constraint(model, g5_def, g5_s == y[2])
    @constraint(model, g5, [g5_s, l[5]] in SOS1())

    @variable(model, g6_s >= 0)
    @constraint(model, g6_def, g6_s == - y[2] + 20)
    @constraint(model, g6, [g6_s, l[6]] in SOS1())

    @variable(model, g7_s >= 0)
    @constraint(model, g7_def, g7_s == x[3] - 0.4*y[3] - 0.7*y[4])
    @constraint(model, g7, [g7_s, l[7]] in SOS1())

    @variable(model, g8_s >= 0)
    @constraint(model, g8_def, g8_s == x[4] - 0.6*y[3] - 0.3*y[4])
    @constraint(model, g8, [g8_s, l[8]] in SOS1())

    @variable(model, g9_s >= 0)
    @constraint(model, g9_def, g9_s == y[3])
    @constraint(model, g9, [g9_s, l[9]] in SOS1())

    @variable(model, g10_s >= 0)
    @constraint(model, g10_def, g10_s == - y[3] + 40)
    @constraint(model, g10, [g10_s, l[10]] in SOS1())

    @variable(model, g11_s >= 0)
    @constraint(model, g11_def, g11_s == y[4])
    @constraint(model, g11, [g11_s, l[11]] in SOS1())

    @variable(model, g12_s >= 0)
    @constraint(model, g12_def, g12_s == -y[4] + 40)
    @constraint(model, g12, [g12_s, l[12]] in SOS1())

    return model
end

model = build_model()
