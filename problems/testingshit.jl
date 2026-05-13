using MarbleBenchmarks, Marble
using AmplNLReader
using MPCCModels
using JuMP, Gurobi

nl_path = "/Users/micahreich/Documents/MarbleBenchmarks/data/macmpec/MacMPEC-nl/trafficsignalcycle-tsc-2.nl"
ampl = AmplModel(nl_path)
ind_ccc2 = findall(!iszero, ampl.meta.cvar)
ind_vcc1 = ampl.meta.cvar[ind_ccc2]
mpcc = MPCCModels.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
data = from_mpcc(mpcc)
jumpmodel = convert(JuMP.Model, data)

set_optimizer(jumpmodel, Gurobi.Optimizer)
optimize!(jumpmodel)
xstar = value.(jumpmodel[:x])

Marble.obj(data, xstar)
norm(Marble.residual_eq(data, xstar), Inf)
norm(Marble.residual_ineq(data, xstar), Inf)
norm(Marble.residual_comp(data, xstar), Inf)

marble_opts = Marble.SolverOptions()
Marble.solve(data; opts=marble_opts)