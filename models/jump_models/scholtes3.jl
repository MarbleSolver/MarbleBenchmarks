# scholtes3.mod	QOR2-MN-2-0
# Original AMPL coding by Sven Leyffer

# An QPEC from S. Scholtes

# Number of variables:   2 slack
# Number of constraints: 0

using JuMP

function build_model()
    model = Model()

    @variable(model, x[1:2] >= 0)

    @objective(model, Min, 0.5*( (x[1] - 1)^2 + (x[2] - 1)^2 ))

    @constraint(model, LCP, [x[1], x[2]] in SOS1())

    return model
end

model = build_model()
