# qpec1.mod	QQR2-MN-v-v
# Original AMPL coding by Sven Leyffer, University of Dundee

# A QPEC from H. Jiang and D. Ralph, Smooth SQP methods for mathematical
# programs with nonlinear complementarity constraints, University of
# Melbourne, December 1997.

# Number of variables:  v
# Number of constraints: v

# ... parameters enlarge the problem
const n = 10
const m = 20
const N = 1:n
const M = 1:m
const NM = (n+1):m

# ... constants
const rr = fill(1.0, n)
const ss = fill(2.0, m)

using JuMP

function build_model()
    model = Model()

    # ... variables
    @variable(model, x[N])
    @variable(model, y[M] >= 0)

    # ... problem statement
    @objective(model, Min, sum((x[i] + rr[i])^2 for i in N) + sum((y[j] + ss[j])^2 for j in M))

    @variable(model, lin1_s[N] >= 0)
    @constraint(model, lin1_def[i in N], lin1_s[i] == y[i] - x[i])
    @constraint(model, lin1[i in N], [lin1_s[i], y[i]] in SOS1())

    @constraint(model, lin2[i in NM], [y[i], y[i]] in SOS1())

    return model
end

model = build_model()
