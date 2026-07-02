include(joinpath(@__DIR__, "..", "Benchmarks.jl"))
using .Benchmarks

const RESULTS_DIR = joinpath(@__DIR__, "results")
const PROBLEMS = list_problems(:macmpec)

bench(f) = f(; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR)

##

df_gurobi = bench(benchmark_gurobi)
df_crisp  = bench(benchmark_crisp)
df_lcqpow = bench(benchmark_lcqpow)
df_marble = bench(benchmark_marble)
