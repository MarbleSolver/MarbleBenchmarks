# bilin.mod	QQR2-MN-8-5
# Original AMPL coding by Sven Leyffer

# An bilevel linear program due to Hansen, Jaumard and Savard,
# "New branch-and-bound rules for linear bilevel programming,
# SIAM J. Sci. Stat. Comp. 13, 1194-1217, 1992. See also book
# Mathematical Programs with Equilibrium Constraints,
# by Luo, Pang & Ralph, CUP, 1997, p. 357.

# Number of variables:   2 + 3 multipliers
# Number of constraints: 4

using JuMP

function build_model()
    model = Model()

    @variable(model, x[1:2] >= 0)

    # ...multipliers
    @variable(model, y[1:6] >= 0)

    @objective(model, Max, 8*x[1] + 4*x[2] - 4*y[1] + 40*y[2] + 4*y[3])

    @constraint(model, lin, x[1] + 2*x[2] - y[3] <= 1.3)

    @variable(model, KKT1_s >= 0)
    @constraint(model, KKT1_def, KKT1_s == 2 - y[4] - 2*y[5] + 4*y[6])
    @constraint(model, KKT1, [KKT1_s, y[1]] in SOS1())

    @variable(model, KKT2_s >= 0)
    @constraint(model, KKT2_def, KKT2_s == 1 + y[4] + 4*y[5] - 2*y[6])
    @constraint(model, KKT2, [KKT2_s, y[2]] in SOS1())

    @variable(model, KKT3_s >= 0)
    @constraint(model, KKT3_def, KKT3_s == 2 + y[4] - y[5] - y[6])
    @constraint(model, KKT3, [KKT3_s, y[3]] in SOS1())

    @variable(model, slack1_s >= 0)
    @constraint(model, slack1_def, slack1_s == 1 + y[1] - y[2] - y[3])
    @constraint(model, slack1, [slack1_s, y[4]] in SOS1())

    @variable(model, slack2_s >= 0)
    @constraint(model, slack2_def, slack2_s == 2 - 4*x[1] + 2*y[1] - 4*y[2] + y[3])
    @constraint(model, slack2, [slack2_s, y[5]] in SOS1())

    @variable(model, slack3_s >= 0)
    @constraint(model, slack3_def, slack3_s == 2 - 4*x[2] - 4*y[1] + 2*y[2] + y[3])
    @constraint(model, slack3, [slack3_s, y[6]] in SOS1())

    return model
end

model = build_model()
