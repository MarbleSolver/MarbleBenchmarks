using MarbleBenchmarks
using Marble
using Plots
const RESULTS_DIR = joinpath(@__DIR__, "results")
const PROBLEMS = [list_problems(:macmpec); list_problems(:robotics)]

"""
    performance_ratios(problems, times) -> Dict solver => Vector{Float64}

`r[p, s] = t[p, s] / min_{s'} t[p, s']`, and `Inf` where solver `s` failed.
"""
function performance_ratios(problems, times)
    best = Dict(p => minimum(times[s][p] for s in keys(times)) for p in problems)
    ratios = Dict{String,Vector{Float64}}()
    for (s, d) in times
        ratios[s] = [isfinite(best[p]) ? d[p] / best[p] : Inf for p in problems]
    end
    return ratios
end


"""
    plot_performance_profile(problems, ratios; tau_max=200.0, n_tau=100, ...) -> Plot

Performance profile in the paper style: % of problems solved vs performance
ratio τ on a log10 x-axis. The profile is sampled on a log-spaced τ grid and
drawn as smooth lines, one per solver.

Sizing/typography knobs (defaults reproduce the current figure):
- `size`: overall figure size in px; enlarge to give the plot more room.
- `tickfontsize`, `guidefontsize`, `legendfontsize`: shrink these to make the
  axis numbers/labels smaller so the axes area takes up more of the figure.
- `margin`: outer whitespace around the axes; lower it to grow the plot area.
"""
function plot_performance_profile(problems, ratios, order; tau_max=500.0, n_tau=100,
        size=(350, 190), tickfontsize=8, guidefontsize=11, legendfontsize=6,
        legend=:bottomright, margin=1.0Plots.mm, kwargs...)
    np = length(problems)
    taus = exp.(range(log(1.0), stop=log(tau_max), length=n_tau))

    plt = plot(; size, legend,
                 xscale=:log10, xlims=(1.0, tau_max), ylims=(0, 105),
                 framestyle=:box, grid=true, gridalpha=0.15,
                 tickfontsize, guidefontsize, legendfontsize, margin,
                 background_color_legend=RGBA(1, 1, 1, 0.9), foreground_color_legend=nothing,
                 kwargs...)
    for s in order
        pct = [count(<=(τ), ratios[s]) for τ in taus] ./ np .* 100
        plot!(plt, taus, pct; label=s, lw=1.4)
    end
    xlabel!(plt, raw"\(10^{x}\) times best")
    ylabel!(plt, "% problems solved")
    return plt
end

MARBLE_OPTIONS = (
    relaxation_min = 1e-10,
    penalty_max = 1e7,
    convergence_eq_violation = 1e-4,
    convergence_ineq_violation = 1e-4,
    convergence_comp_violation = 1e-4,
    inertia_warmstart = false,
    comp_init_random = false,
    relaxation_scaling = 0.25,
    retraction_type = Marble.Softplus,
    clamp_hessian=true,
)

times = Dict{String,Dict{String,Float64}}()

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; MARBLE_OPTIONS...)
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["zero"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.z .= randn(length(solver.workspace.z))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["z_randn"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.s_ineq .= randn(length(solver.workspace.s_ineq))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["ineq_randn"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.s_comp .= randn(length(solver.workspace.s_comp))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["comp_randn"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.z .= randn(length(solver.workspace.z))  # warmstart with all ones
    solver.workspace.s_ineq .= randn(length(solver.workspace.s_ineq))  # warmstart with all ones
    solver.workspace.s_comp .= randn(length(solver.workspace.s_comp))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["all_randn"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.z .= ones(length(solver.workspace.z))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["z_ones"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.s_ineq .= ones(length(solver.workspace.s_ineq))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["ineq_ones"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.s_comp .= ones(length(solver.workspace.s_comp))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["comp_ones"] = d

d = Dict{String,Float64}(String(p) => Inf for p in PROBLEMS); for prob in PROBLEMS
    mpcc = MarbleBenchmarks.Benchmarks.get_mpcc(prob)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; warmstart = false, MARBLE_OPTIONS...)
    solver.workspace.z .= ones(length(solver.workspace.z))  # warmstart with all ones
    solver.workspace.s_ineq .= ones(length(solver.workspace.s_ineq))  # warmstart with all ones
    solver.workspace.s_comp .= ones(length(solver.workspace.s_comp))  # warmstart with all ones
    result = Marble.solve!(solver)
    @assert result.converged
    d[String(prob)] = result.solve_time_s
end
times["all_ones"] = d

ratios = performance_ratios([String(p) for p in PROBLEMS], times)

order = ["zero", "z_randn", "ineq_randn", "comp_randn", "all_randn"]
plot_performance_profile([String(p) for p in PROBLEMS], ratios, order; tau_max=10.0, dpi=300, xticks = 10 .^ collect(0:0.2:1))

order = ["zero", "z_ones", "ineq_ones", "comp_ones", "all_ones"]
plot_performance_profile([String(p) for p in PROBLEMS], ratios, order; tau_max=10.0, dpi=300, xticks = 10 .^ collect(0:0.2:1))


