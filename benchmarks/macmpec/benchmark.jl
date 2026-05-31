using CCOpt, MadNLP, MadNLPHSL
using JuMP, Gurobi
using Marble, MarbleBenchmarks
using NLPModels, AmplNLReader
using LinearAlgebra, DataFrames, CSV

include(joinpath(@__DIR__, "../common.jl"))
include(joinpath(@__DIR__, "../benchmark_ccopt.jl"))
include(joinpath(@__DIR__, "../benchmark_gurobi.jl"))
include(joinpath(@__DIR__, "../benchmark_marble.jl"))

results_dir = joinpath(@__DIR__, "results")

## CCOpt
benchmark_ccopt(problems=list_problems(:macmpec); save=true, results_dir=results_dir)

## Gurobi
benchmark_gurobi(problems=list_problems(:macmpec); save=false, results_dir=results_dir)

## Marble
df = benchmark_marble(problems=list_problems(:macmpec); save=false, results_dir=results_dir)