using CSV, DataFrames, Statistics, Printf
using GLMakie

const ROOT_DIR = @__DIR__
const RESULTS_DIRS = [joinpath(ROOT_DIR, "results"), ROOT_DIR]
const FIGURES_DIR = joinpath(ROOT_DIR, "figures")
mkpath(FIGURES_DIR)

const SUCCESS = Set(["OPTIMAL", "LOCALLY_SOLVED", "SOLVE_SUCCEEDED"])
const EPS_TIME = 1e-12
const MIN_GAP = 1e-14

function latest_result_file(solver::String)
    prefix = "$(solver)_results_"
    files = String[]

    for dir in RESULTS_DIRS
        isdir(dir) || continue
        append!(
            files,
            joinpath.(Ref(dir), filter(f -> startswith(f, prefix) && endswith(f, ".csv"), readdir(dir)))
        )
    end

    isempty(files) && error("No results for '$solver' in $(join(RESULTS_DIRS, ", "))")
    return sort(files; by = basename)[end]
end

function load_latest(solver::String)
    path = latest_result_file(solver)
    @info "[$solver] loading $(basename(path))"

    df = CSV.read(path, DataFrame)

    required = [
        :problem,
        :status,
        :obj,
        :residual_eq,
        :residual_ineq,
        :residual_comp,
        :total_time_s,
    ]

    missing_cols = setdiff(required, propertynames(df))
    isempty(missing_cols) || error("$(basename(path)) missing columns: $missing_cols")

    return select(df, required)
end

is_success(s) = !ismissing(s) && strip(String(s)) in SUCCESS
is_finite_num(x) = !ismissing(x) && isfinite(x)
is_pos_time(x) = is_finite_num(x) && x > 0

function keep_valid_successes!(df::DataFrame)
    filter!(
        r -> is_success(r.status) &&
             is_finite_num(r.obj) &&
             is_pos_time(r.total_time_s) &&
             is_finite_num(r.residual_comp),
        df,
    )
    return df
end

raw_g = keep_valid_successes!(load_latest("gurobi"))
raw_c = keep_valid_successes!(load_latest("ccopt"))
raw_m = keep_valid_successes!(load_latest("marble"))

df = innerjoin(
    select(raw_g, :problem,
           :total_time_s  => :t_gurobi,
           :obj           => :obj_gurobi,
           :residual_comp => :comp_gurobi),
    select(raw_c, :problem,
           :total_time_s  => :t_ccopt,
           :obj           => :obj_ccopt,
           :residual_comp => :comp_ccopt),
    select(raw_m, :problem,
           :total_time_s  => :t_marble,
           :obj           => :obj_marble,
           :residual_comp => :comp_marble),
    on = :problem,
)

n = nrow(df)
n > 0 || error("No common successfully solved problems across all solvers.")
@info "Common problems across all solvers: $n"

ref_obj = abs.(df.obj_gurobi) .+ 1e-10
df[!, :gap_ccopt]  = abs.(df.obj_ccopt  .- df.obj_gurobi) ./ ref_obj
df[!, :gap_marble] = abs.(df.obj_marble .- df.obj_gurobi) ./ ref_obj

const C_GUROBI = RGBf(0.20, 0.47, 0.87)
const C_CCOPT  = RGBf(0.91, 0.38, 0.13)
const C_MARBLE = RGBf(0.13, 0.67, 0.40)

set_theme!(
    theme_light();
    fontsize = 13,
    Axis = (
        spinewidth     = 0.8,
        xgridvisible   = true,
        ygridvisible   = true,
        gridcolor      = (:black, 0.09),
        gridwidth      = 0.5,
        xticklabelsize = 11,
        yticklabelsize = 11,
    ),
)

# ---------------------------------------------------------------------
# Figure 1: Dolan-Moré performance profile
# ---------------------------------------------------------------------

sort!(df, :t_gurobi)

T = Matrix{Float64}(df[:, [:t_gurobi, :t_ccopt, :t_marble]])
T = max.(T, EPS_TIME)

best = minimum(T; dims = 2)
R = T ./ best

τ_max = max(1.05, maximum(R) * 1.05)
τ_vals = exp10.(range(0.0, log10(τ_max); length = 1000))
ρ = [count(R[:, s] .<= τ) / n for τ in τ_vals, s in 1:3]

fig1 = Figure(size = (680, 400))
ax1 = Axis(fig1[1, 1];
    title  = "Performance Profile - MacMPEC Benchmark",
    xlabel = "Performance ratio (t_solver / t_best)",
    ylabel = "Fraction of problems solved",
    xscale = log10,
    yticks = 0:0.1:1.0,
)

for (i, (label, color)) in enumerate(zip(
        ("Gurobi", "CCOpt", "Marble"),
        (C_GUROBI, C_CCOPT, C_MARBLE)))
    lines!(ax1, τ_vals, ρ[:, i]; color, label, linewidth = 2.4)
end

hlines!(ax1, [1.0]; color = (:black, 0.15), linestyle = :dot, linewidth = 1.0)
axislegend(ax1; position = :rb, framevisible = true, labelsize = 12)
ylims!(ax1, -0.02, 1.08)

save(joinpath(FIGURES_DIR, "1_perf_profile.png"), fig1; px_per_unit = 2)
display(fig1)

# ---------------------------------------------------------------------
# Figure 2: CCOpt vs Marble solve time
# ---------------------------------------------------------------------

t_ccopt = max.(df.t_ccopt, EPS_TIME)
t_marble = max.(df.t_marble, EPS_TIME)
t_gurobi = max.(df.t_gurobi, EPS_TIME)

t_all = vcat(t_ccopt, t_marble)
t_lo = minimum(t_all) / 1.4
t_hi = maximum(t_all) * 1.4

fig2 = Figure(size = (560, 520))
ax2 = Axis(fig2[1, 1];
    title  = "CCOpt vs. Marble Solve Time",
    xlabel = "CCOpt solve time (s)",
    ylabel = "Marble solve time (s)",
    xscale = log10,
    yscale = log10,
    aspect = 1,
)

lines!(ax2, [t_lo, t_hi], [t_lo, t_hi];
    color = (:black, 0.35), linestyle = :dash, linewidth = 1.1)

sc = scatter!(ax2, t_ccopt, t_marble;
    color       = log10.(t_gurobi),
    colormap    = :viridis,
    markersize  = 11,
    strokecolor = :black,
    strokewidth = 0.6,
)

Colorbar(fig2[1, 2], sc;
    label         = "log10(Gurobi time, s)",
    labelsize     = 11,
    ticklabelsize = 10,
    width         = 14,
)

for row in eachrow(df)
    ratio = row.t_ccopt / row.t_marble
    
    text!(ax2, max(row.t_ccopt, EPS_TIME) * 1.15, max(row.t_marble, EPS_TIME);
        text     = String(row.problem),
        fontsize = 8,
        align    = (:left, :center),
    )
end

xlims!(ax2, t_lo, t_hi)
ylims!(ax2, t_lo, t_hi)

save(joinpath(FIGURES_DIR, "2_ccopt_vs_marble_time.png"), fig2; px_per_unit = 2)
display(fig2)

# ---------------------------------------------------------------------
# Figure 3: objective gap vs Gurobi
# ---------------------------------------------------------------------

gap_sort_order = sortperm(max.(df.gap_ccopt, df.gap_marble); rev = true)
df_gap = df[gap_sort_order, :]

gap_c = max.(df_gap.gap_ccopt, MIN_GAP)
gap_m = max.(df_gap.gap_marble, MIN_GAP)

probs_gap = String.(df_gap.problem)
pidx_gap = collect(1:nrow(df_gap))

fig3 = Figure(size = (680, max(540, nrow(df_gap) * 22 + 100)))
ax3 = Axis(fig3[1, 1];
    title     = "Objective Gap vs. Gurobi Global Solution",
    xlabel    = "|f_solver - f*| / (|f*| + eps)  [Gurobi = reference]",
    yticks    = (pidx_gap, probs_gap),
    xscale    = log10,
    yreversed = true,
)

vlines!(ax3, 1e-2; color = (:red, 0.35), linestyle = :dash, linewidth = 1.1)

text!(ax3, 1e-2 * 1.15, 0.4;
    text     = "1% gap",
    fontsize = 9,
    color    = RGBAf(0.7, 0.1, 0.1, 0.8),
    align    = (:left, :center),
)

scatter!(ax3, gap_c, pidx_gap .- 0.2;
    color = (C_CCOPT, 0.9),
    marker = :rect,
    markersize = 9,
    label = "CCOpt",
    strokecolor = :black,
    strokewidth = 0.5,
)

scatter!(ax3, gap_m, pidx_gap .+ 0.2;
    color = (C_MARBLE, 0.9),
    marker = :diamond,
    markersize = 10,
    label = "Marble",
    strokecolor = :black,
    strokewidth = 0.5,
)

axislegend(ax3; position = :rb, framevisible = true, labelsize = 12)

save(joinpath(FIGURES_DIR, "3_obj_gap.png"), fig3; px_per_unit = 2)
display(fig3)

# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

println()
println("="^62)
println("  MacMPEC Benchmark Summary   ($n problems, all solvers converged)")
println("="^62)

println("\n  Solve Time (ms):")
@printf("  %-8s  %8s  %8s  %8s\n", "Solver", "Median", "Mean", "Max")
for (name, col) in (
        ("Gurobi", df.t_gurobi),
        ("CCOpt",  df.t_ccopt),
        ("Marble", df.t_marble))
    @printf("  %-8s  %8.3f  %8.3f  %8.3f\n",
        name, median(col) * 1e3, mean(col) * 1e3, maximum(col) * 1e3)
end

println("\n  Speedup vs. Gurobi:")
@printf("  %-8s  %10s  %10s  %10s\n", "Solver", "Median×", "GeoMean×", "Max×")
for (name, col) in (
        ("CCOpt",  df.t_gurobi ./ max.(df.t_ccopt, EPS_TIME)),
        ("Marble", df.t_gurobi ./ max.(df.t_marble, EPS_TIME)))
    gm = exp(mean(log.(max.(col, EPS_TIME))))
    @printf("  %-8s  %10.2f  %10.2f  %10.2f\n",
        name, median(col), gm, maximum(col))
end

println("\n  Objective gap vs. Gurobi:")
@printf("  %-8s  %12s  %12s\n", "Solver", "Median gap", "Max gap")
for (name, col) in (
        ("CCOpt",  df.gap_ccopt),
        ("Marble", df.gap_marble))
    @printf("  %-8s  %12.2e  %12.2e\n",
        name, median(col), maximum(col))
end

println("\n  Problems with gap > 1%:")
found_any = false
for row in eachrow(df)
    for (name, gap, fval) in (
            ("CCOpt",  row.gap_ccopt,  row.obj_ccopt),
            ("Marble", row.gap_marble, row.obj_marble))
        if gap > 0.01
            @printf("    %-12s  %-8s  gap = %.2e  (f=%.6g  vs  f*=%.6g)\n",
                row.problem, name, gap, fval, row.obj_gurobi)
            found_any = true
        end
    end
end
found_any || println("    (none)")

println("\n  Figures saved to: $FIGURES_DIR")
println()