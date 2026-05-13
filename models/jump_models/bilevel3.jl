# bilevel3.mod	QQR2-MN-16-12
# Original AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 11

# Number of variables:   12
# Number of constraints: 11
# Nonlinear complementarity constraint

using JuMP

function build_model()
    model = Model()

    @variable(model, x[1:2] >= 0)
    @variable(model, y[1:6])
    @variable(model, l[1:4])

    @objective(model, Min, - x[1]^2 - 3*x[2] - 4*y[1] + y[2]^2)

    @constraint(model, c1, x[1]^2 + 2*x[2] <= 4)
    @constraint(model, F1, 0 == 2*y[1] + 2*y[3] - 3*y[4] - y[5])
    @constraint(model, F2, 0 == - 5 - y[3] + 4*y[4] - y[6])
    @constraint(model, F3, 0 == x[1]^2 - 2*x[1] + x[2]^2 - 2*y[1] + y[2] + 3 - ( l[1] ))
    @constraint(model, F4, 0 == x[2] + 3*y[1] - 4*y[2] - 4      - ( l[2] ))
    @constraint(model, F5, 0 == y[1]                            - ( l[3] ))
    @constraint(model, F6, 0 == y[2]                            - ( l[4] ))

    @variable(model, g1_s >= 0)
    @constraint(model, g1_def, g1_s == l[1])
    @variable(model, g1_t >= 0)
    @constraint(model, g1_tdef, g1_t == y[3])
    @constraint(model, g1, [g1_s, g1_t] in SOS1())

    @variable(model, g2_s >= 0)
    @constraint(model, g2_def, g2_s == l[2])
    @variable(model, g2_t >= 0)
    @constraint(model, g2_tdef, g2_t == y[4])
    @constraint(model, g2, [g2_s, g2_t] in SOS1())

    @variable(model, g3_s >= 0)
    @constraint(model, g3_def, g3_s == l[3])
    @variable(model, g3_t >= 0)
    @constraint(model, g3_tdef, g3_t == y[5])
    @constraint(model, g3, [g3_s, g3_t] in SOS1())

    @variable(model, g4_s >= 0)
    @constraint(model, g4_def, g4_s == l[4])
    @variable(model, g4_t >= 0)
    @constraint(model, g4_tdef, g4_t == y[6])
    @constraint(model, g4, [g4_s, g4_t] in SOS1())

    return model
end

model = build_model()
