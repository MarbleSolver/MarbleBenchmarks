using Plots
pgfplotsx()

# Window (first quadrant is where the complementarity variables live, s,t ≥ 0)
const LO, HI = -0.15, 1.2

fmtκ(κ) = rstrip(rstrip(string(round(κ; digits = 3)), '0'), '.')

# First-quadrant branch of the relaxed complementarity curve s·t = κ, clipped to
# the window (NaN off-window so the line breaks cleanly).
function hyperbola(κ; n = 500)
    ss = range(max(LO, κ / HI), HI; length = n)
    ts = κ ./ ss
    ts[(ts .> HI) .| (ts .< LO)] .= NaN
    return ss, ts
end

# Feasible portion of the curve: the arc with s + t ≤ 1. On s·t = κ we have
# s + t ≤ 1 exactly for s ∈ [(1-d)/2, (1+d)/2] with d = √(1-4κ); it is empty
# once κ > 0.25 (the curve clears the band entirely).
function feasible_arc(κ; n = 200)
    κ > 0.25 && return (Float64[], Float64[])
    d = sqrt(1 - 4κ)
    ss = range((1 - d) / 2, (1 + d) / 2; length = n)
    return ss, κ ./ ss
end

# The lone tangent solution, present only at the fold κ = 0.25.
tangent_point(κ) = isapprox(κ, 0.25; atol = 1e-9) ? [(0.5, 0.5)] : Tuple{Float64,Float64}[]

const BAND = RGBA(0.45, 0.55, 0.75, 0.22)   # shaded inequality region
const GOLD = RGB(0.96, 0.70, 0.09)          # highlighted feasible solutions
# curve colour ramp: κ = 0.1 (in the band) → 0.25 (tangent) → 0.4 (outside)
const KAPPA_CMAP = cgrad([RGB(0.07, 0.19, 0.43), RGB(0.15, 0.50, 0.82), RGB(0.55, 0.62, 0.70)])

function solnchange_plot(; κs = [0.1, 0.25, 0.4])
    n = length(κs)
    colof(i) = n == 1 ? KAPPA_CMAP[0.5] : KAPPA_CMAP[(i - 1) / (n - 1)]

    plt = plot(; size = 0.8 .* (560, 540), aspect_ratio = 1, legend = :topright,
        xlims = (LO, HI), ylims = (LO, HI),
        xticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        yticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        xlabel = raw"\(s\)", ylabel = raw"\(t\)",
        framestyle = :box, grid = true, gridalpha = 0.15,
        background_color_legend = RGBA(1, 1, 1, 0.9), foreground_color_legend = nothing)

    # Shaded inequality region -1 ≤ s+t ≤ 1 (only the s+t ≤ 1 edge falls in view).
    band = Shape([(LO, LO), (1 - LO, LO), (LO, 1 - LO)])
    plot!(plt, band; lw = 1.0, linecolor = BAND, fillcolor = BAND, label = raw"\(-1\le s+t\le 1\)")

    # Relaxed complementarity curves, one per κ.
    for (i, κ) in enumerate(κs)
        plot!(plt, hyperbola(κ)...; color = colof(i), lw = 2.2,
            label = "\\(s\\cdot t = $(fmtκ(κ))\\)")
    end

    # Highlighted feasible solution set, plus the lone tangent solution.
    for κ in κs
        fs, ft = feasible_arc(κ)
        isempty(fs) || plot!(plt, fs, ft; color = GOLD, lw = 5, label = "")
    end
    for κ in κs
        pts = tangent_point(κ)
        isempty(pts) || scatter!(plt, first.(pts), last.(pts);
            marker = :circle, ms = 6, color = GOLD,
            markerstrokecolor = :white, markerstrokewidth = 1.2, label = "")
    end
    # Legend key for the highlight (drawn off-window).
    plot!(plt, [2.0, 2.0], [2.0, 2.0]; color = GOLD, lw = 5, label = "feasible set")
    return plt
end

plt = solnchange_plot()
savefig(plt, joinpath(@__DIR__, "solnchange.svg"))
display(plt)
