# ralph2.mod	QUR-AN-LCP-2-0-1
# Original AMPL coding by Sven Leyffer

# An LPEC from D. Ralph, Judge Inst., University of Cambridge.

using JuMP

function build_model()
    model = Model()

    @variable(model, x >= 0)
    @variable(model, y >= 0)

    @objective(model, Min, x^2 + y^2 - 4*x*y)

    @constraint(model, compl, [x, y] in SOS1())

    return model
end

model = build_model()
