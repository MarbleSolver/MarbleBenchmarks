# bard3m.mod	QQR2-MN-
# Original AMPL coding by Sven Leyffer

# From GAMS model in mpeclib of Steven Dirkse, see
# http://www1.gams.com/mpec/mpeclib.htm

using JuMP

function build_model()
    model = Model()

    # ... upper level variables x
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    # .. lower level variables
    @variable(model, y1 >= 0)
    @variable(model, y2 >= 0)
    @variable(model, m_cons1 >= 0)
    @variable(model, m_cons2 >= 0)

    # .. upper level problem objective
    @objective(model, Min, -x1^2 - 3*x2 + y2^2 - 4*y1)

    @constraint(model, side, x1^2 + 2*x2 <= 4)

    # ... optimality conditions of lower level problem
    @variable(model, cons1_s >= 0)
    @constraint(model, cons1_def, cons1_s == x1^2 - 2*x1 + x2^2 - 2*y1 + y2 + 3)
    @constraint(model, cons1, [cons1_s, m_cons1] in SOS1())

    @variable(model, cons2_s >= 0)
    @constraint(model, cons2_def, cons2_s == x2 + 3*y1 - 4*y2 - 4)
    @constraint(model, cons2, [cons2_s, m_cons2] in SOS1())

    @variable(model, d_y1_s >= 0)
    @constraint(model, d_y1_def, d_y1_s == (2*y1+2*m_cons1)-3*m_cons2)
    @constraint(model, d_y1, [d_y1_s, y1] in SOS1())

    @variable(model, d_y2_s >= 0)
    @constraint(model, d_y2_def, d_y2_s == (-5-m_cons1)+4*m_cons2)
    @constraint(model, d_y2, [d_y2_s, y2] in SOS1())

    return model
end

model = build_model()
