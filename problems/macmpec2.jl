using AmplNLReader
using NLPModels
using JuMP
using LinearAlgebra
using MarbleBenchmarks

NL_PATH = "/Users/micahreich/Documents/MarbleBenchmarks/data/macmpec/MacMPEC-nl/liswet1-inv-liswet1-200.nl"

ampl = AmplModel(NL_PATH)
jump = to_jump(ampl)

## using Gurobi
using Gurobi
set_optimizer(jump, Gurobi.Optimizer)
optimize!(jump)
