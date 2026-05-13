# bard1m.mod	QQR2-MN-9-5
# Original AMPL coding by Sven Leyffer

# From GAMS model in mpeclib of Steven Dirkse, see
# http://www1.gams.com/mpec/mpeclib.htm

using JuMP

function build_model()
    model = Model()

    @variable(model, x >= 0)
    @variable(model, y >= 0)

    @variable(model, sy >= 0)
    @variable(model, l[1:3] >= 0)

    @objective(model, Min, (x-5)^2 + (2*y + 1)^2)

    @variable(model, cons1_s >= 0)
    @constraint(model, cons1_def, cons1_s == 3*x - y - 3)
    @constraint(model, cons1, [cons1_s, l[1]] in SOS1())

    @variable(model, cons2_s >= 0)
    @constraint(model, cons2_def, cons2_s == -x + 0.5*y + 4)
    @constraint(model, cons2, [cons2_s, l[2]] in SOS1())

    @variable(model, cons3_s >= 0)
    @constraint(model, cons3_def, cons3_s == -x - y + 7)
    @constraint(model, cons3, [cons3_s, l[3]] in SOS1())

    @constraint(model, d_y, sy == (((2*(y-1)-1.5*x)-l[1]*(-1)*1)-l[2]*0.5)-l[3]*(-1)*1)

    return model
end

model = build_model()
