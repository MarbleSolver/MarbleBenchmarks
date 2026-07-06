using MarbleBenchmarks

const RESULTS_DIR = joinpath(@__DIR__, "results")
const PROBLEMS = list_problems(:macmpec)

##

df_gurobi = benchmark_gurobi(; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR)
df_crisp  = benchmark_crisp(; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR)
df_lcqpow = benchmark_lcqpow(; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR)
df_marble = benchmark_marble(; problems=PROBLEMS, save=true, results_dir=RESULTS_DIR)
