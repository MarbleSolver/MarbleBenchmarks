using Plots
pgfplotsx()

const WIN = (-0.12, 1.12)                         # data window
const PAD = 0.16                                  # extra top/right room
const HI  = WIN[2] + PAD                          # axis upper limit

clip(v, lo = WIN[1], hi = WIN[2]) = lo ≤ v ≤ hi ? v : NaN   # drop off-window points

# boundary-curve samples (s, t), NaN off-window so lines break
function curve1(κ; n = 400)                        # s·t = κ
    κ == 0 && return ([0.0], [2.0])
    ss = range(κ / HI, HI, length = n)
    return ss, κ ./ ss
end
function curve2(κ; n = 400)                        # (1-s)(1-t) = κ
    κ == 0 && return ([1.0, 1.0, NaN, -0.5, 2.0], [-0.5, 2.0, NaN, 1.0, 1.0])
    ss = range(WIN[1], 1 - κ / WIN[2], length = n)
    return ss, clip.(1 .- κ ./ (1 .- ss))
end

# feasible points: intersections of the two boundaries
function feasible_points(κ)
    κ > 0.25 && return Tuple{Float64,Float64}[]
    d = sqrt(1 - 4κ)
    return unique([((1 + d) / 2, (1 - d) / 2), ((1 - d) / 2, (1 + d) / 2)])
end

fmtκ(κ) = rstrip(rstrip(string(round(κ; digits = 3)), '0'), '.')

# one colour per κ (navy → blue → green); s·t = κ solid, (1-s)(1-t) = κ dashed
const KAPPA_CMAP = cgrad([RGB(0.07, 0.19, 0.43), RGB(0.15, 0.50, 0.82), RGB(0.18, 0.64, 0.30)])

# κs use the colour ramp; κs_off (past the fold) are faded gray
function bifurcation_plot(; κs = [1e-7, 0.1, 0.25], κs_off = [0.55])
    n = length(κs)
    colof(i) = n == 1 ? KAPPA_CMAP[0.5] : KAPPA_CMAP[(i - 1) / (n - 1)]
    OFF = RGBA(0.4, 0.4, 0.4, 0.5)

    plt = plot(; size = 0.75 .* (560, 520), aspect_ratio = 1, legend = :topright,
        xlims = (WIN[1], HI), ylims = (WIN[1], HI),
        xticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        yticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        xlabel = raw"\(s\)", ylabel = raw"\(t\)",
        framestyle = :box, grid = true, gridalpha = 0.18,
        top_margin = 3Plots.mm, right_margin = 3Plots.mm,
        background_color_legend = RGBA(1, 1, 1, 0.9), foreground_color_legend = nothing)

    # feasible locus: s + t = 1
    plot!(plt, [WIN[1], WIN[2]], [WIN[2], WIN[1]]; color = :gray75, lw = 1, ls = :dot, label = "")

    function draw_kappa!(κ, col)
        s1, t1 = curve1(κ); s2, t2 = curve2(κ)
        plot!(plt, s1, t1; color = col, lw = 2.6, label = "")             # s·t = κ
        plot!(plt, s2, t2; color = col, lw = 2.6, ls = :dot, label = "")  # (1-s)(1-t) = κ
        pts = feasible_points(κ)
        isempty(pts) || scatter!(plt, first.(pts), last.(pts);
            marker = :rect, ms = 7, color = col,
            markerstrokecolor = :white, markerstrokewidth = 1.2, label = "")
    end

    # legend keys, drawn off-window
    key = ([2.0, 2.0], [2.0, 2.0])
    plot!(plt, key...; color = :gray20, lw = 2.4, label = raw"\(s t = \kappa\)")
    plot!(plt, key...; color = :gray20, lw = 2.4, ls = :dot, label = raw"\((1-s)(1-t) = \kappa\)")
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
