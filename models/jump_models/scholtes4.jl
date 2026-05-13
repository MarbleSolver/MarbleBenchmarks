# scholtes4.mod	LQR2-MN-3-2
# Original AMPL coding by Sven Leyffer

# An LPEC from S. Scholtes, Judge Inst., University of Cambridge.

# Number of variables:   3 slack
# Number of constraints: 2

using JuMP

function build_model()
    model = Model()

    @variable(model, z[1:2] >= 0)
    @variable(model, z3)

    @objective(model, Min, z[1] + z[2] - z3)

    @constraint(model, lin1, -4 * z[1] + z3 <= 0)
    @constraint(model, lin2, -4 * z[2] + z3 <= 0)

    @constraint(model, compl, [z[1], z[2]] in SOS1())

    return model
end

model = build_model()
