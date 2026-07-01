using CCOpt, MadNLP, MadNLPHSL
using JuMP, Gurobi
using CRISP
using Marble, MarbleBenchmarks
using NLPModels, AmplNLReader
using LinearAlgebra, DataFrames, CSV

include(joinpath(@__DIR__, "../common.jl"))
include(joinpath(@__DIR__, "../benchmark_ccopt.jl"))
include(joinpath(@__DIR__, "../benchmark_gurobi.jl"))
include(joinpath(@__DIR__, "../benchmark_marble.jl"))
include(joinpath(@__DIR__, "../benchmark_crisp.jl"))

results_dir = joinpath(@__DIR__, "results")

## CCOpt
df_ccopt = benchmark_ccopt(problems=list_problems(:macmpec); save=true, results_dir=results_dir)

## Gurobi
df_gurobi = benchmark_gurobi(problems=list_problems(:macmpec); save=true, results_dir=results_dir)

## CRISP
df_crisp = benchmark_crisp(problems=list_problems(:macmpec); save=true, results_dir=results_dir)

## Marble

df_marble = benchmark_marble(problems=list_problems(:macmpec); save=true, results_dir=results_dir)

# ## individual
# name = Symbol("liswet1-200")
# mpcc = get_problem(name)

# solver = Marble.Solver()
# Marble.setup!(solver, mpcc;
#                 verbosity=1,
#                 max_iters=2000,
#                 relax_min=1e-8,
#                 penalty_max=1e9,
#                 relax_scaling=0.1,
#                 use_relax_correction=true,
#                 convergence_eq_violation=1e-8,
#                 convergence_ineq_violation=1e-8,
#                 convergence_comp_violation=1e-8
# )
# result = Marble.solve!(solver)

# @info """
# obj: $(Marble.obj(solver, Marble.z(result)))
# factorizations: $(Marble.factorizations(result))
# factorizations ldlt: $(Marble.factorizations_ldlt(result))
# factorizations inertia: $(Marble.factorizations_inertia(result))
# factorizations linesearch: $(Marble.factorizations_linesearch(result))
# """