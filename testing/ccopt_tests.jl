using CCOpt, MadNLP, MadNLPHSL
using ADNLPModels
using NLPModels

## Model 1
f(x) = x' * x

c(x) = [
    x[1] + x[2]
]

x0 = randn(2)

lvar = [0.0, 0.0]        # x[1] >= 0, x[2] >= 0
uvar = [Inf, Inf]

lcon = [0.0]             # x[1] + x[2] = 0
ucon = [0.0]

nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

mpcc = CCOpt.MPCCModelVarVar(nlp, [1], [2])

## Model 2
f(x) = x' * x

c(x) = [
    x[1];
    2 - x[1];
]

lvar = [0.0]
uvar = [Inf]

lcon = [0.5, 0.0]             # x[1] + x[2] = 0
ucon = [Inf, Inf]

nlp = ADNLPModel(f, [0.0], lvar, uvar, c, lcon, ucon)

mpcc = CCOpt.MPCCModelVarCon(nlp, [1], [2])

##

vmpcc = CCOpt.vertical_form(mpcc)

opts = CCOpt.RelaxationOptions()
madnlpc_solver_options = Dict(
    :bound_relax_factor=>0.0,
    :bound_push=>1e-1,
    :print_level=>MadNLP.ERROR,
    :max_iter=>4000,
    :linear_solver=>Ma27Solver,
    :verbose=>true,
)

solver = CCOpt.RelaxationSolver(vmpcc; solver_opts=opts, madnlpc_solver_options...)
stats = CCOpt.solve_homotopy!(solver)