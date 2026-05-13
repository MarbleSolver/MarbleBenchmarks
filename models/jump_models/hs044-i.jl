# hs044-i.mod	QLR2-MY-20-14-10
# Original AMPL coding by Sven Leyffer

# A QPEC obtained by varying the rhs of HS44 (a QP)
# from an idea communicated by S. Scholtes.

using JuMP

function build_model(;
    sol = Dict(1=>0, 2=>3, 3=>0, 4=>4),
    g   = Dict(1=>1, 2=>-1, 3=>-1, 4=>0),
    A   = Dict((1,1)=>-1,(1,2)=>-2,(2,1)=>-4,(2,2)=>-1,
               (3,1)=>-3,(3,2)=>-4,
               (4,3)=>-2,(4,4)=>-1,
               (5,3)=>-1,(5,4)=>-2,
               (6,3)=>-1,(6,4)=>-1),
    b   = Dict(1=>8, 2=>12, 3=>12, 4=>8, 5=>8, 6=>5),
    H   = Dict((1,3)=>-1,(1,4)=>1,(2,3)=>1,(2,4)=>-1,
               (3,1)=>-1,(3,2)=>1,(4,1)=>1,(4,2)=>-1),
    zl  = Dict(1=>0.01, 2=>-10.0, 3=>0.1, 4=>-1.0, 5=>-1.0, 6=>0.001),
    zu  = Dict(1=>10.0, 2=>-0.01, 3=>1.0, 4=>-0.1, 5=>1.0,  6=>10.0),
    u   = Dict(1=>0.2, 2=>1.2, 3=>2.0, 4=>0.1, 5=>0.1, 6=>-0.1),
    v   = Dict(1=>1.2, 2=>0.2, 3=>0.1, 4=>2.0, 5=>10.0, 6=>-0.2))

    I = 1:4
    J = 1:6
    K = 1:6

    model = Model()

    # ... variables (states, i.e. primal/dual pair of HS44)
    @variable(model, x[I] >= 0)
    @variable(model, l[J] >= 0)
    @variable(model, m[I] >= 0)

    # ... variables (controls, i.e. perturbations to rhs & g)
    @variable(model, z[k in K], lower_bound=zl[k], upper_bound=zu[k])

    # ... minimize l_2 norm of derivation from optimal solution
    @objective(model, Min, sum((sol[i] - x[i])^2 for i in I))

    # ... perturbed KKT conditions (1st order)
    @constraint(model, KKT[i in I],
        sum(get(H, (i,ii), 0)*x[ii] for ii in I) + (g[i] + u[i]*z[i])
        - sum(get(A, (j,i), 0)*l[i] for j in J) - m[i] == 0)

    # ... perturbed slackness conditions (general c/s)
    @variable(model, slackness_g_s[J] >= 0)
    @constraint(model, slackness_g_def[j in J],
        slackness_g_s[j] == (b[j] - v[j]*z[j]) + sum(get(A, (j,i), 0)*x[i] for i in I))
    @constraint(model, slackness_g[j in J], [l[j], slackness_g_s[j]] in SOS1())

    # ... perturbed slackness conditions (simple bounds)
    @constraint(model, slackness_x[i in I], [x[i], m[i]] in SOS1())

    return model
end

model = build_model()
