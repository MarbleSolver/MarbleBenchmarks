# stackelberg1.mod	QQR2-MN-4-2
# Original AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 6

# Number of variables:   3
# Number of constraints: 2

using JuMP

function build_model()
    model = Model()

    @variable(model, 0 <= x <= 200)
    @variable(model, y >= 0)
    @variable(model, l >= 0)

    @objective(model, Min, 0.5*x^2 + 0.5*x*y - 95*x)

    @constraint(model, F, 2*y + 0.5*x - 100 - l == 0)
    @constraint(model, g, [y, l] in SOS1())

    return model
end

model = build_model()
