module Benchmarks

using LinearAlgebra
using DataFrames, CSV, Dates
using JuMP
using Marble
using ..MarbleBenchmarks   # parent module: get_problem, list_problems, …
using Gurobi
using CRISP, LCQPow
using CCOpt, MadNLP, MadNLPHSL

const MOI = JuMP.MOI

include("common.jl")
include("solvers/marble.jl")
include("solvers/gurobi.jl")
include("solvers/crisp.jl")
include("solvers/lcqpow.jl")

export benchmark_marble, benchmark_gurobi, benchmark_crisp, benchmark_lcqpow, benchmark_ccopt
export get_problem, list_problems, problem_classes
export run_benchmark, SolverResult, get_mpcc, feasibility_stats

end
