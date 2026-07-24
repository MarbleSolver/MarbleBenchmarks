using MarbleBenchmarks
using Marble
const PROBLEMS = list_problems(:robotics)

MARBLE_OPTIONS = (
    max_iters = 2000,
    relaxation_min = 1e-6,
    penalty_max = 1e6,
    inertia_warmstart = false,
    comp_init_random = false,
    clamp_hessian = true,
    relaxation_scaling = 0.5,
    verbosity = 0,
)

using ProgressMeter
sizes = 10:10:100
n_samples = 100
times = zeros(length(sizes), n_samples)
for (i, N) in enumerate(sizes)
    println("Running problem size N = $N")
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(:hopper_2d, N = N)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; MARBLE_OPTIONS...)
    for j in 1:n_samples
        result = Marble.solve!(solver)
        times[i, j] = result.solve_time_s
    end
end

# Plot mean and std of solve times vs problem size
using Plots, Statistics
mean_times = mean(times, dims=2)
std_times = std(times, dims=2)
p1 = plot(sizes, mean_times, ribbon=std_times, xlabel="Horizon (N)", ylabel="Solve Time (s)", title="Hopper Solve Time vs Horizon", legend=false)

# 
const RESULTS_DIR = joinpath(@__DIR__, "results")
const PROBLEMS = list_problems(:macmpec)
df_marble = Benchmarks.run_benchmark("marble_softplus", Benchmarks.marble_solve; problems=PROBLEMS, save=false, options=MARBLE_OPTIONS)
probs = [MarbleBenchmarks.Benchmarks.get_mpcc(Symbol(p)) for p in df_marble.problem]
p2 = scatter(df_marble.total_time_s, [prob.nlp.meta.nvar for prob in probs], xlabel="Total Time (s)", ylabel="Number of Variables", title="Solver Performance vs Problem Size", legend=false, xscale=:log10, yscale=:log10)

plot(p2, p1, layout=(1, 2), size=(1200, 400))
