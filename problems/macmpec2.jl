using AmplNLReader
using JSON

_macmpec_qpcc = begin
    path = joinpath(@__DIR__, "..", "data", "macmpec", "macmpec_qpcc.json")
    js = JSON.parsefile(path)

    filter(k -> js[k]["convex_cost"] && js[k]["linear_constraints"], collect(keys(js))) |> sort
end

##

using AmplNLReader
using NLPModels
using JuMP
using LinearAlgebra
using Marble

NL_PATH = "/Users/micahreich/Documents/MarbleBenchmarks/data/macmpec/MacMPEC-nl/bard1.nl"

ampl = AmplModel(NL_PATH)
jump = to_jump(ampl)
nlp, ind_vcc1, ind_vcc2 = to_vertical_mpcc(jump)

# data = MarbleData(nlp, ind_vcc1, ind_vcc2)
# Marble.solve(data; opts=Marble.SolverOptions())

# ## using Gurobi
# using Gurobi
# set_optimizer(jump, Gurobi.Optimizer)
# optimize!(jump)
