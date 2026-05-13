# portfl-i.mod	QLR2-AY-NLP-87-25-12
# AMPL coding by Sven Leyffer, Feb. 2001,
# from portfl models of Bob Vanderbei.

# A QPEC obtained by varying the grad of portfl1-6
# from an idea communicated by S. Scholtes.
#
# This is almost a sensible problem. Here, ask what
# vector r of minimum norm perturbations to returns R,
# gives a solution that is as close as possible to the
# given solution (obtained by rounding soln to portfl1-6).
#
# Problem has data files portfl1.dat - portfl6.dat with
# different parameters F & R.

using JuMP

function build_model(F, R, sol)
    NS = 12
    NR = 62

    model = Model()

    @variable(model, s[1:NS] >= 0.0)
    @variable(model, m_var[1:NS] >= 0.0)
    @variable(model, l)
    @variable(model, r[1:NR] >= 0)

    @objective(model, Min,
        sum((s[i] - sol[i])^2 for i in 1:NS)
        + sum(r[i]^2 for i in 1:NR))

    @constraint(model, KKT[k in 1:NS],
        0 == sum(2*(sum(s[j]*F[j,i] for j in 1:NS) - (R[i]+r[i]))*F[k,i] for i in 1:NR)
             - l - m_var[k])

    @constraint(model, cons1, sum(s[i] for i in 1:NS) == 1)

    @constraint(model, compl_s[i in 1:NS], [s[i], m_var[i]] in SOS1())

    return model
end
