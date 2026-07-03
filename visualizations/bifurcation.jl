# visualizations/bifurcation.jl
#
# How the feasible set of TWO complementarity constraints changes as the
# (Scholtes) relaxation parameter κ grows — a minimal analogue of Fig. 2.
#
# Variables (s, t). Two complementarities, each relaxed as  product = κ:
#     0 ≤ s    ⊥ t    ≥ 0   ⇝   s·t        = κ     (boundary, solid)
#     0 ≤ 1-s  ⊥ 1-t  ≥ 0   ⇝   (1-s)(1-t) = κ     (boundary, dashed)
#
# The two boundary curves meet where (expand the 2nd, substitute s·t = κ)
#     s + t = 1     and     s(1-s) = κ   ⇒   s = (1 ± √(1-4κ)) / 2 .
# So there are TWO feasible points for κ < 1/4; they slide along s + t = 1 toward
# the centre and collide into ONE at κ = 1/4 (a fold). This is the simplest
# bifurcation of the feasible set of two complementarities.

using Plots
pgfplotsx()   # same backend as benchmarks/macmpec/analysis.jl (LaTeX-quality output)

const WIN = (-0.12, 1.12)                         # data window on both axes
const PAD = 0.16                                  # extra room on the top and right
const HI  = WIN[2] + PAD                          # axis upper limit; curves span out to here

clip(v, lo = WIN[1], hi = WIN[2]) = lo ≤ v ≤ hi ? v : NaN   # drop points off-window

# Boundary-curve samples (s, t) inside the window (NaN elsewhere so lines break).
function curve1(κ; n = 400)                        # s·t = κ  (s, t ≥ 0 branch for κ > 0)
    κ == 0 && return ([0.0], [2.0])  # full lines s=0, t=0
    ss = range(κ / HI, HI, length = n)             # span to the top/right frame (no cut-off)
    return ss, κ ./ ss
end
function curve2(κ; n = 400)                        # (1-s)(1-t) = κ  (s, t ≤ 1 branch for κ > 0)
    κ == 0 && return ([1.0, 1.0, NaN, -0.5, 2.0], [-0.5, 2.0, NaN, 1.0, 1.0])  # full lines s=1, t=1
    ss = range(WIN[1], 1 - κ / WIN[2], length = n)
    return ss, clip.(1 .- κ ./ (1 .- ss))
end

# The complementarity-feasible points (intersections of the two boundaries).
function feasible_points(κ)
    κ > 0.25 && return Tuple{Float64,Float64}[]
    d = sqrt(1 - 4κ)
    return unique([((1 + d) / 2, (1 - d) / 2), ((1 - d) / 2, (1 + d) / 2)])
end

fmtκ(κ) = rstrip(rstrip(string(round(κ; digits = 3)), '0'), '.')

# One colour per κ (sequential navy → blue → green). Line style encodes the
# constraint: s·t = κ solid, (1-s)(1-t) = κ dashed.
const KAPPA_CMAP = cgrad([RGB(0.07, 0.19, 0.43), RGB(0.15, 0.50, 0.82), RGB(0.18, 0.64, 0.30)])

# `κs` are drawn with the colour ramp; `κs_off` (past the fold, no feasible
# point) are drawn in a faded gray to set them apart.
function bifurcation_plot(; κs = [1e-7, 0.1, 0.25], κs_off = [0.6])
    n = length(κs)
    colof(i) = n == 1 ? KAPPA_CMAP[0.5] : KAPPA_CMAP[(i - 1) / (n - 1)]
    OFF = RGBA(0.4, 0.4, 0.4, 0.5)   # faded gray for the post-bifurcation κ

    plt = plot(; size = 0.75 .* (560, 520), aspect_ratio = 1, legend = :topright,
        xlims = (WIN[1], HI), ylims = (WIN[1], HI),
        xticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        yticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        xlabel = raw"\(s\)", ylabel = raw"\(t\)",
        framestyle = :box, grid = true, gridalpha = 0.18,
        top_margin = 3Plots.mm, right_margin = 3Plots.mm,
        background_color_legend = RGBA(1, 1, 1, 0.9), foreground_color_legend = nothing)

    # Locus of every feasible point: the line s + t = 1.
    plot!(plt, [WIN[1], WIN[2]], [WIN[2], WIN[1]]; color = :gray75, lw = 1, ls = :dot, label = "")

    # One κ's two boundary curves (+ its feasible points) in a given colour.
    function draw_kappa!(κ, col)
        s1, t1 = curve1(κ); s2, t2 = curve2(κ)
        plot!(plt, s1, t1; color = col, lw = 2.6, label = "")              # s·t = κ         (solid)
        plot!(plt, s2, t2; color = col, lw = 2.6, ls = :dash, label = "")  # (1-s)(1-t) = κ  (dashed)
        pts = feasible_points(κ)
        isempty(pts) || scatter!(plt, first.(pts), last.(pts);
            marker = :rect, ms = 7, color = col,
            markerstrokecolor = :white, markerstrokewidth = 1.2, label = "")
    end

    # Legend keys (off-window, clipped points so they render on pgfplotsx):
    # neutral solid/dashed explain the two constraints, colours explain κ.
    key = ([2.0, 2.0], [2.0, 2.0])
    plot!(plt, key...; color = :gray20, lw = 2.4, label = raw"\(s t = \kappa\)")
    plot!(plt, key...; color = :gray20, lw = 2.4, ls = :dash, label = raw"\((1-s)(1-t) = \kappa\)")
    for (i, κ) in enumerate(κs)
        plot!(plt, key...; color = colof(i), lw = 2.4, label = "\\(\\kappa = $(fmtκ(κ))\\)")
    end
    for κ in κs_off
        plot!(plt, key...; color = OFF, lw = 2.4, label = "\\(\\kappa = $(fmtκ(κ))\\)")
    end

    for (i, κ) in enumerate(κs); draw_kappa!(κ, colof(i)); end
    for κ in κs_off; draw_kappa!(κ, OFF); end
    return plt
end

plt = bifurcation_plot()
savefig(plt, joinpath(@__DIR__, "bifurcation.svg"))
display(plt)
