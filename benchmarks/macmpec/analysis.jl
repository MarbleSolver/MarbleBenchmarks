using CSV, DataFrames, Printf
using Plots
pgfplotsx()   # PGFPlotsX backend so figures save as standalone .tex (TikZ/pgfplots)

const RESULTS_DIR = joinpath(@__DIR__, "results")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

# A solve counts as "solved" iff its recorded feasibility residuals are all
# within this tolerance. This is the single place that decision is made
const FEAS_TOL = 1e-7

is_solved(row; tol = FEAS_TOL) =
    isfinite(row.total_time_s) &&
    row.residual_eq   <= tol &&
    row.residual_ineq <= tol &&
    row.residual_comp <= tol

# Display name and line color per solver (unknown solvers fall back sensibly).
const SOLVER_LABEL = Dict(
    "marble" => "Marble (ours)",
    "lcqpow" => "LCQPow",
    "gurobi" => "Gurobi",
    "crisp"  => "CRISP",
)
const SOLVER_COLOR = Dict(
    "marble" => RGB(0.90, 0.13, 0.13),   # bright red
    "lcqpow" => RGB(0.15, 0.50, 0.82),   # bright blue
    "gurobi" => RGB(0.18, 0.72, 0.45),   # bright green
    "crisp"  => RGB(0.61, 0.35, 0.82),   # bright purple
)
# Draw order: these solvers first (in this order), then any others alphabetically.
const SOLVER_ORDER = ["marble", "lcqpow", "crisp", "gurobi",]

solver_label(s) = get(SOLVER_LABEL, s, uppercasefirst(s))
solver_color(s) = get(SOLVER_COLOR, s, :auto)
function ordered_solvers(ratios)
    present = collect(keys(ratios))
    known  = [s for s in SOLVER_ORDER if s in present]
    extras = sort([s for s in present if s ∉ SOLVER_ORDER])
    return vcat(known, extras)
end

"""
    load_latest_results(dir) -> Dict{String,DataFrame}

Read the summary CSVs in `dir` (the `stats/` subfolder of the results dir), one
per solver. Filenames look like `marble_2026-07-02_17-45-02.csv`; the solver name
is the prefix before the first `_`. If several files exist for a solver, keep the
most recent (the timestamped names sort chronologically).
"""
function load_latest_results(dir=joinpath(RESULTS_DIR, "stats"))
    files = filter(f -> endswith(f, ".csv"), readdir(dir))
    isempty(files) && error("no result CSVs found in $dir")
    latest = Dict{String,String}()
    for f in sort(files)                       # ascending → newest overwrites
        solver = String(split(f, '_')[1])
        latest[solver] = f
    end
    return Dict(solver => CSV.read(joinpath(dir, f), DataFrame)
                for (solver, f) in latest)
end

"""
    solver_metrics(results; metric=:total_time_s) -> (problems, times)

`problems` is the sorted union of problem names. `times[solver][problem]` is the
metric value when `is_solved(row)` (feasibility residuals within `FEAS_TOL`),
else `Inf`.
"""
function solver_metrics(results; metric=:total_time_s)
    problems = String[]
    for df in values(results)
        append!(problems, String.(df.problem))
    end
    problems = sort(unique(problems))

    times = Dict{String,Dict{String,Float64}}()
    for (solver, df) in results
        d = Dict{String,Float64}(p => Inf for p in problems)
        for row in eachrow(df)
            if is_solved(row)
                d[String(row.problem)] = Float64(getproperty(row, metric))
            end
        end
        times[solver] = d
    end
    return problems, times
end

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
function plot_performance_profile(problems, ratios; tau_max=500.0, n_tau=100,
        size=(350, 190), tickfontsize=8, guidefontsize=11, legendfontsize=8,
        margin=1.0Plots.mm)
    np = length(problems)
    taus = exp.(range(log(1.0), stop=log(tau_max), length=n_tau))

    plt = plot(; size, legend=:bottomright,
                 xscale=:log10, xlims=(1.0, tau_max), ylims=(0, 105),
                 framestyle=:box, grid=true, gridalpha=0.15,
                 tickfontsize, guidefontsize, legendfontsize, margin,
                 background_color_legend=RGBA(1, 1, 1, 0.9), foreground_color_legend=nothing)
    for s in ordered_solvers(ratios)
        pct = [count(<=(τ), ratios[s]) for τ in taus] ./ np .* 100
        plot!(plt, taus, pct; label=solver_label(s), color=solver_color(s), lw=1.4)
    end
    xlabel!(plt, raw"\(10^x\) times best")
    ylabel!(plt, "% problems solved")
    return plt
end

"""
    print_summary(problems, ratios; thresholds)

Text table: fraction of problems each solver solves within τ× best.
"""
function print_summary(problems, ratios; thresholds = [1, 2, 4, 8, 10, 16, 100])
    np = length(problems)
    solvers = sort(collect(keys(ratios)))
    println("\nFraction of $np problems solved within τ× the best (fastest) time:")
    @printf("  %-8s", "τ")
    foreach(s -> @printf("%-12s", s), solvers)
    println()
    for τ in sort(thresholds)
        @printf("  %-8g", τ)
        for s in solvers
            @printf("%-12.2f", count(<=(float(τ)), ratios[s]) / np)
        end
        println()
    end

    # Worst-case slowdown: the largest performance ratio over the problems each
    # solver actually solved (failed problems have Inf ratio and are excluded).
    println("\nWorst-case slowdown (max τ over problems the solver solved):")
    for s in solvers
        finite = filter(isfinite, ratios[s])
        if isempty(finite)
            @printf("  %-8s  %s\n", s, "— (no problems solved)")
        else
            @printf("  %-8s  %.2f×\n", s, maximum(finite))
        end
    end
end

"""
    print_global_summary(results; ref="gurobi", rtol=1e-3, atol=1e-4)

How many problems each solver reaches the *global* optimum on: it must return a
feasible solution (`is_solved`) whose objective matches the reference solver's
(`ref` = Gurobi, which solves the MacMPEC MPCCs to global optimality) within
`|Δobj| ≤ atol + rtol·|obj_ref|`. Denominator is the set of problems `ref` solves.
"""
function print_global_summary(results; ref="gurobi", rtol=1e-3, atol=1e-4)
    haskey(results, ref) || (@warn "no `$ref` results; skipping global-solution summary"; return)
    obj    = Dict(s => Dict(String(r.problem) => Float64(r.obj) for r in eachrow(df)) for (s, df) in results)
    solved = Dict(s => Dict(String(r.problem) => is_solved(r)   for r in eachrow(df)) for (s, df) in results)
    gref = Dict(p => obj[ref][p] for p in keys(obj[ref]) if solved[ref][p])   # global objective per problem
    nref = length(gref)
    matches(a, b) = abs(a - b) <= atol + rtol * abs(b)

    println("\nGlobal solutions found — objective matches $ref (of the $nref problems $ref solves globally):")
    for s in ordered_solvers(results)
        s == ref && continue
        hits = count(p -> get(solved[s], p, false) && haskey(obj[s], p) && matches(obj[s][p], gref[p]), keys(gref))
        @printf("  %-8s %2d / %d  (%.1f%%)\n", s, hits, nref, 100 * hits / nref)
    end
end

# Save `plt` as a TikZ `.tex`. Plots' pgfplotsx backend hard-codes an absolute
# mm size, which overflows a narrow paper column. Wrapping the picture in
# \resizebox{\linewidth}{!}{…} makes it scale to the enclosing column width, so
# it never runs off the page (requires \usepackage{graphicx} in the document).
function save_tex(plt, path)
    savefig(plt, path)
    tex = read(path, String)
    tex = replace(tex, "\\begin{tikzpicture}" => "\\resizebox{\\linewidth}{!}{%\n\\begin{tikzpicture}"; count = 1)
    tex = replace(tex, "\\end{tikzpicture}"   => "\\end{tikzpicture}%\n}"; count = 1)
    write(path, tex)
    return path
end

function main(; metric = :total_time_s, tau_max = 1000.0, plot_kwargs...)
    results = load_latest_results()
    problems, times = solver_metrics(results; metric)
    ratios = performance_ratios(problems, times)

    mkpath(FIGURES_DIR)
    plt = plot_performance_profile(problems, ratios; tau_max, plot_kwargs...)
    display(plt)
    out = joinpath(FIGURES_DIR, "performance_profile.svg")
    save_tex(plt, out)
    @info "wrote $out"

    print_summary(problems, ratios)
    print_global_summary(results)
    return problems, ratios
end

main()
