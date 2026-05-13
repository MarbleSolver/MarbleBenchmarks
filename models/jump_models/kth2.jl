# kth2.mod
#
# simple MPEC # 2

using JuMP

function build_model()
    model = Model()

    @variable(model, z1 >= 0)
    @variable(model, z2 >= 0)

    @objective(model, Min, z1 + (z2 - 1)^2)

    @constraint(model, compl, [z1, z2] in SOS1())

    return model
end

model = build_model()
