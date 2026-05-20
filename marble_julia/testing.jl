using Marble
using MarbleBenchmarks

include("solver.jl")
include("../benchmarks/common.jl")

opts_exact = LDQPCCOptions(
    method = :exact,
    # method = :gn,
    μ0 = 1.0,
    κ0 = 1.0,
    μ_min = 1e-8,
    κ_min = 1e-8,
    reduction = 0.2,
    verbose = true,
    # reg = 1e-6,
)

(nlp, ind_cc1, ind_cc2), cc_types = macmpec_to_mpcc("sl1"), (:var, :con)
data = MarbleData(nlp, ind_cc1, ind_cc2, cc_types)

res = solve_ld_qpcc(data; opts=opts_exact);

norm(data.J_eq*res.x + data.b_eq, Inf)
norm(min.(0, data.J_ineq*res.x + data.b_ineq), Inf)
norm((data.L*res.x + data.l) .* (data.R*res.x + data.r), Inf)
res.objective