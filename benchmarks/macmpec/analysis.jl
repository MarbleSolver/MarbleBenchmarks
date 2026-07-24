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
    "marble_softplus" => "Marble (softplus)",
    "marble_exp"      => "Marble (exp)",
    "lcqpow"          => "LCQPow",
    "gurobi"          => "Gurobi",
    "crisp"           => "CRISP",
)
const SOLVER_COLOR = Dict(
    "marble_softplus" => RGB(0.90, 0.13, 0.13),   # bright red
    "marble_exp"      => RGB(0.95, 0.52, 0.10),   # bright orange
    "lcqpow"          => RGB(0.15, 0.50, 0.82),   # bright blue
    "gurobi"          => RGB(0.18, 0.72, 0.45),   # bright green
    "crisp"           => RGB(0.61, 0.35, 0.82),   # bright purple
)
# Draw order: these solvers first (in this order), then any others alphabetically.
const SOLVER_ORDER = ["marble_softplus", "marble_exp", "lcqpow", "crisp", "gurobi",]
# Result CSVs for these solvers stay on disk but are left out of the analysis.
# `marble` is the pre-split run, superseded by `marble_softplus`/`marble_exp`.
const SKIP_SOLVERS = Set(["marble"])

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
per solver. Filenames look like `marble_softplus_2026-07-02_17-45-02.csv`; the
solver name is everything before the trailing `_<date>_<time>` stamp, so solver
names may themselves contain `_`. If several files exist for a solver, keep the
most recent (the timestamped names sort chronologically). Solvers in
`SKIP_SOLVERS` are ignored.
"""
function load_latest_results(dir=joinpath(RESULTS_DIR, "stats"))
    files = filter(f -> endswith(f, ".csv"), readdir(dir))
    isempty(files) && error("no result CSVs found in $dir")
    latest = Dict{String,String}()
    for f in sort(files)                       # ascending → newest overwrites
        parts  = split(splitext(f)[1], '_')    # …, "2026-07-02", "17-45-02"
        solver = String(join(parts[1:end-2], '_'))
        solver in SKIP_SOLVERS && continue
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
    solvers = ordered_solvers(ratios)
    println("\nFraction of $np problems solved within τ× the best (fastest) time:")
    @printf("  %-8s", "τ")
    foreach(s -> @printf("%-20s", solver_label(s)), solvers)
    println()
    for τ in sort(thresholds)
        @printf("  %-8g", τ)
        for s in solvers
            @printf("%-20.2f", count(<=(float(τ)), ratios[s]) / np)
        end
        println()
    end

    # Worst-case slowdown: the largest performance ratio over the problems each
    # solver actually solved (failed problems have Inf ratio and are excluded),
    # so the count it is taken over is printed alongside it.
    println("\nWorst-case slowdown (max τ over the problems the solver solved):")
    for s in solvers
        finite = filter(isfinite, ratios[s])
        if isempty(finite)
            @printf("  %-20s %s\n", solver_label(s), "— (no problems solved)")
        else
            @printf("  %-20s %7.2f×  (over %2d / %d solved)\n",
                    solver_label(s), maximum(finite), length(finite), np)
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
        @printf("  %-20s %2d / %d  (%.1f%%)\n", solver_label(s), hits, nref, 100 * hits / nref)
    end
end

"""
    plot_solved_and_global(results; ref="gurobi", rtol=1e-3, atol=1e-4, ...) -> Plot

Grouped bar chart, one group per solver: the % of all problems it solves and the
% of `ref`'s (Gurobi's) globally-solved problems on which it also reaches the
global optimum (`|Δobj| ≤ atol + rtol·|obj_ref|`). The global bars are annotated
with the raw count, so the chart shows both "how well" and "how many global".
"""
function plot_solved_and_global(results; ref="gurobi", rtol=1e-3, atol=1e-4,
        highlight="marble_softplus", size=(480, 250), tickfontsize=7, guidefontsize=10,
        legendfontsize=7, annfontsize=6, legend=:bottomright, margin=1.0Plots.mm, kwargs...)
    solvers = ordered_solvers(results)

    problems = sort(unique(reduce(vcat, [String.(df.problem) for df in values(results)])))
    np = length(problems)
    solved = Dict(s => Dict(String(r.problem) => is_solved(r) for r in eachrow(df)) for (s, df) in results)
    pct_solved = Dict(s => 100 * count(values(solved[s])) / np for s in solvers)

    # global-optimum hits vs the reference solver (Gurobi solves MacMPEC globally)
    have_ref = haskey(results, ref)
    have_ref || @warn "no `$ref` results; global-solution bars omitted"
    obj = Dict(s => Dict(String(r.problem) => Float64(r.obj) for r in eachrow(df)) for (s, df) in results)
    global_hits = Dict{String,Int}()
    nref = 0
    if have_ref
        gref = Dict(p => obj[ref][p] for p in keys(obj[ref]) if solved[ref][p])
        nref = length(gref)
        matches(a, b) = abs(a - b) <= atol + rtol * abs(b)
        for s in solvers
            global_hits[s] = count(p -> get(solved[s], p, false) && haskey(obj[s], p) && matches(obj[s][p], gref[p]), keys(gref))
        end
    end
    pct_global = Dict(s => (have_ref && nref > 0) ? 100 * global_hits[s] / nref : 0.0 for s in solvers)

    n = length(solvers)
    xs = collect(1:n)
    w = 0.38
    SOLVED_COL = RGB(0.20, 0.45, 0.70)
    GLOBAL_COL = RGB(0.95, 0.60, 0.15)
    FADE = 0.5   # the reference solver (Gurobi) is drawn semi-transparent so it recedes

    plt = plot(; size, legend, legend_columns=-1,
        framestyle=:box, grid=true, gridalpha=0.15,
        ylims=(0, 112), xlims=(0.4, n + 0.6),
        xticks=(xs, [s == highlight ? "\\textbf{$(solver_label(s))}" : solver_label(s) for s in solvers]), xrotation=25,
        tickfontsize, guidefontsize, legendfontsize, margin,
        # legend inside the box, bottom-right; wider gap between the two entries
        extra_kwargs=Dict(:subplot => Dict("legend style" =>
            raw"{at={(0.985,0.03)}, anchor=south east, draw=black, fill=white, fill opacity=0.9, text opacity=1, /tikz/every even column/.append style={column sep=0.6cm}}")),
        kwargs...)

    solved_vals = [pct_solved[s] for s in solvers]
    global_vals = [pct_global[s] for s in solvers]

    # full-opacity bars for every solver except the reference (masked to NaN),
    # then draw the reference (Gurobi) faded so the other solvers stand out
    solved_base = [s == ref ? NaN : v for (s, v) in zip(solvers, solved_vals)]
    global_base = [s == ref ? NaN : v for (s, v) in zip(solvers, global_vals)]
    bar!(plt, xs .- w / 2, solved_base; bar_width=w,
        color=SOLVED_COL, linewidth=0, label="\\% problems solved")
    bar!(plt, xs .+ w / 2, global_base; bar_width=w,
        color=GLOBAL_COL, linewidth=0, label="\\% global")
    if have_ref
        ri = findfirst(==(ref), solvers)
        bar!(plt, [ri - w / 2], [solved_vals[ri]]; bar_width=w,
            color=RGBA(SOLVED_COL, FADE), linewidth=0, label="")
        bar!(plt, [ri + w / 2], [global_vals[ri]]; bar_width=w,
            color=RGBA(GLOBAL_COL, FADE), linewidth=0, label="")
    end

    # value labels: percentage above each bar
    for (i, s) in enumerate(solvers)
        annotate!(plt, i - w / 2, pct_solved[s] + 4,
            text(@sprintf("%.0f", pct_solved[s]), annfontsize, SOLVED_COL, :center))
        have_ref && annotate!(plt, i + w / 2, pct_global[s] + 4,
            text(@sprintf("%.0f", pct_global[s]), annfontsize, GLOBAL_COL, :center))
    end
    ylabel!(plt, "\\%")
    return plt
end

"""
    plot_combined(problems, ratios, results; ...) -> Plot

The performance profile and the solved/global bar chart side by side in a single
figure. Rendering both in one pgfplots (1×2) layout is what makes their black
axis boxes exactly the same height: with equal `top`/`bottom` margins on the two
subplots, Plots gives them equal-height plot areas and each subplot's own
legend / axis-label furniture is drawn outside that area. (Two separate files
can't guarantee this, because each box's height then depends on its own
furniture.) Keep the two margins equal across subplots and the boxes stay
matched. Include this one file in the paper if equal box heights matter.
"""
function plot_combined(problems, ratios, results; tau_max=1000.0, size=(900, 300),
        top=2.0Plots.mm, bottom=1.0Plots.mm)
    # SAME top/bottom margins on both subplots ⇒ identical black-box heights
    p1 = plot_performance_profile(problems, ratios; tau_max, top_margin=top, bottom_margin=bottom)
    p2 = plot_solved_and_global(results; top_margin=top, bottom_margin=bottom)
    return plot(p1, p2; layout=(1, 2), size)
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

    barplt = plot_solved_and_global(results)
    display(barplt)
    barout = joinpath(FIGURES_DIR, "solved_global.svg")
    save_tex(barplt, barout)
    @info "wrote $barout"

    comb = plot_combined(problems, ratios, results; tau_max)
    display(comb)
    combout = joinpath(FIGURES_DIR, "combined.svg")
    save_tex(comb, combout)
    @info "wrote $combout"

    print_summary(problems, ratios)
    print_global_summary(results)
    return problems, ratios
end

main()
