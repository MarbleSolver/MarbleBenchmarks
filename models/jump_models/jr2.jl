# jr2.mod   QLR2-AN-LCP-2-1-1
#
# QPEC from ideas by Jiang & Ralph
#

using JuMP

function build_model()
    model = Model()

    @variable(model, z1)
    @variable(model, z2 >= 0)

    @objective(model, Min, (z2 - 1)^2 + z1^2)

    @variable(model, compl_s >= 0)
    @constraint(model, compl_def, compl_s == z2 - z1)
    @constraint(model, compl, [z2, compl_s] in SOS1())

    return model
end

model = build_model()
