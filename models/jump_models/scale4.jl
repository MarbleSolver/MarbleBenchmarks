# scale4.mod
#

using JuMP

function build_model(; a=100)
    model = Model()

    @variable(model, x1)
    @variable(model, x2 >= 0)

    @objective(model, Min, (a*x1 - 1)^2 + (a*x2 - 1)^2)

    @variable(model, compl_s >= 0)
    @constraint(model, compl_def, compl_s == x1)
    @constraint(model, compl, [compl_s, x2] in SOS1())

    return model
end

model = build_model()
