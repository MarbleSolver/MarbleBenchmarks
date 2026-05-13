# MPEC due to Danny Ralph, see also
#
#   G Maier, F Giannessi and A Nappi, Indirect
#   identification of yield limits by mathematical
#   programming, Engineering Structures 4, 1982,
#   86-98.
#
# From a GAMS model by S.P. Dirkse & M.C. Ferris (MPECLIB),
# (see http://www.gams.com/mpec/).
#
# AMPL coding Sven Leyffer, University of Dundee, Jan. 2000

using JuMP

function build_model(P, M_param, q_param, c_param)
    design = 1:4
    state  = 5:104

    model = Model()

    @variable(model, 0 <= x[design] <= 1000)
    @variable(model, y[state] >= 0)
    @variable(model, s[state] >= 0)

    @objective(model, Min,
        0.5 * (
            sum(x[i] * sum(get(P, (i,j), 0.0) * x[j] for j in design) for i in design)
            + 2.0 * sum(x[i] * sum(get(P, (i,j), 0.0) * y[j] for j in state) for i in design)  # NOTE: AMPL source has y[i] here (likely typo); translated as y[j]
            + sum(y[i] * sum(get(P, (i,j), 0.0) * y[j] for j in state) for i in state)
        )
        + sum(get(c_param, i, 0.0) * x[i] for i in design)
        + sum(get(c_param, i, 0.0) * y[i] for i in state)
    )

    @variable(model, F_s[state] >= 0)
    @constraint(model, F_def[i in state],
        F_s[i] == sum(get(M_param, (i,k), 0.0) * x[k] for k in design)
                + sum(get(M_param, (i,j), 0.0) * y[j] for j in state)
                + get(q_param, i, 0.0))
    @constraint(model, F[i in state], [F_s[i], y[i]] in SOS1())

    return model
end
