using CCOpt
using AmplNLReader

name = "hs044-i"
nl_file = joinpath(@__DIR__, "..", "data", "macmpec", "ampl_nl", name * ".nl")

@assert isfile(nl_file) "NL file not found: $nl_file"

model = AmplModel(nl_file)
ind_ccc2 = findall(model.meta.cvar .!== 0)
ind_vcc1 = model.meta.cvar[ind_ccc2]

mpcc = CCOpt.MPCCModelVarCon(model, ind_vcc1, ind_ccc2)
solver = CCOpt.RelaxationSolver(CCOpt.vertical_form(mpcc))
stats = CCOpt.solve_homotopy!(solver)