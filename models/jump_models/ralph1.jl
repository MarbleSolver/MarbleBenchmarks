# ralph1.mod	LUR-AN-LCP-2-0-1
# Original AMPL coding by Sven Leyffer

# An LPEC from D. Ralph, Judge Inst., University of Cambridge.
# This problem violates strong stationarity, but is B-stationary.

using JuMP

function build_model()
    model = Model()

    @variable(model, x >= 0)
    @variable(model, y >= 0)

    @objective(model, Min, 2*x - y)

    @variable(model, compl_s >= 0)
    @constraint(model, compl_def, compl_s == y - x)
    @constraint(model, compl, [y, compl_s] in SOS1())

    return model
end

model = build_model()
