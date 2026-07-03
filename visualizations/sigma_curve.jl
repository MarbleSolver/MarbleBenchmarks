# visualizations/sigma_curve.jl
#
# The feasible set of a single relaxed complementarity  s·t = κ  is a curve
# (hyperbola). A point on it is parametrised by σ ∈ ℝ and can slide along the
# curve in either direction. Drawn for κ = 0, 0.25, 0.5, 1; the κ = 0.5 curve is
# highlighted (black) with a marked point σ and a double arrow showing it slides
# either way along the curve.

using Plots
pgfplotsx()   # LaTeX-quality output, same backend as the other visualizations

# Colours consistent with bifurcation.jl
const GRAY = RGB(0.4, 0.4, 0.4)
const BLUE = RGB(0.15, 0.50, 0.82)

const XLO, XHI = -0.03, 1.13
const YLO, YHI = -0.03, 1.16

# s·t = κ, positive branch, spanning the window (κ = 0 is the positive axes).
function st_curve(κ; n = 400)
    κ == 0 && return ([0.0, 0.0, NaN, 0.0, XHI], [0.0, YHI, NaN, 0.0, 0.0])
    ss = range(κ / YHI, XHI, length = n)          # t runs from the top frame down
    return ss, κ ./ ss
end

# Small filled-triangle arrowhead with its tip at (px, py) pointing along (dx, dy).
function arrowhead!(plt, px, py, dx, dy; len = 0.06, col = :black)
    n = sqrt(dx^2 + dy^2)
    ux, uy = dx / n, dy / n                       # unit tangent (points to the tip)
    vx, vy = -uy, ux                              # perpendicular
    w = 0.5 * len
    plot!(plt, Shape([px, px - len * ux + w * vx, px - len * ux - w * vx],
                     [py, py - len * uy + w * vy, py - len * uy - w * vy]);
        fillcolor = col, linecolor = col, lw = 0.5, label = "")
    return plt
end

function sigma_curve_plot()
    plt = plot(; size = 0.75 .* (560, 520), legend = false, aspect_ratio = 1,
        xlims = (XLO, XHI), ylims = (YLO, YHI),
        xticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        yticks = ([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        xlabel = raw"\(s\)", ylabel = raw"\(t\)",
        framestyle = :box, grid = true, gridalpha = 0.15,
        top_margin = 3Plots.mm, right_margin = 3Plots.mm)

    # Context: contours of z = s·t (each level is a curve s·t = κ), labelled
    # along the curve by Plots' built-in contour labeller.
    xs = range(0, XHI, length = 200)
    ys = range(0, YHI, length = 200)
    Z = [x * y for y in ys, x in xs]
    contour!(plt, xs, ys, Z; levels = [0.0, 0.08, 0.16], color = cgrad([GRAY, GRAY]),
        clabels = true, colorbar = false, lw = 1)
    # Highlighted curve κ = κ0, drawn along its σ-parametrisation
    #   s(σ) = √κ0 · exp(σ),  t(σ) = √κ0 · exp(-σ)   (so s·t = κ0 for every σ ∈ ℝ).
    # Two marked points at σ = ∓0.1 (green / blue) with a green→blue gradient
    # between them; the curve continues each way with the opacity fading out.
    κ0 = 0.25
    GREEN = RGB(0.18, 0.64, 0.30)
    σ1, σ2 = -0.5, 0.5
    scur(σ) = sqrt(κ0) * exp(σ)
    tcur(σ) = sqrt(κ0) * exp(-σ)
    σmin = log(sqrt(κ0) / YHI)                    # t reaches the top frame
    σmax = log(XHI / sqrt(κ0))                    # s reaches the right frame

    lerp(a, b, u) = a + (b - a) * u
    function σcolor(σ)
        if σ < σ1                                 # negative side: green, fading out
            a = clamp((σ - σmin) / (σ1 - σmin), 0.0, 1.0)
            return RGBA(GREEN.r, GREEN.g, GREEN.b, a)
        elseif σ > σ2                             # positive side: blue, fading out
            a = clamp((σmax - σ) / (σmax - σ2), 0.0, 1.0)
            return RGBA(BLUE.r, BLUE.g, BLUE.b, a)
        else                                      # between the two points: green → blue
            u = (σ - σ1) / (σ2 - σ1)
            return RGBA(lerp(GREEN.r, BLUE.r, u), lerp(GREEN.g, BLUE.g, u), lerp(GREEN.b, BLUE.b, u), 1.0)
        end
    end

    σs = range(σmin, σmax, length = 90)
    for i in 1:length(σs)-1
        σm = (σs[i] + σs[i+1]) / 2
        plot!(plt, [scur(σs[i]), scur(σs[i+1])], [tcur(σs[i]), tcur(σs[i+1])];
            color = σcolor(σm), lw = 3.4, label = "")
    end

    # The two marked σ points, each labelled with its value.
    scatter!(plt, [scur(σ1)], [tcur(σ1)]; marker = :circle, ms = 7.5, color = GREEN,
        markerstrokecolor = :white, markerstrokewidth = 1.5, label = "")
    scatter!(plt, [scur(σ2)], [tcur(σ2)]; marker = :circle, ms = 7.5, color = BLUE,
        markerstrokecolor = :white, markerstrokewidth = 1.5, label = "")
    annotate!(plt, scur(σ1) + 0.04, tcur(σ1) + 0.09, text(raw"\(\sigma = -0.5\)", 11, GREEN, :left))
    annotate!(plt, scur(σ2) + 0.04, tcur(σ2) + 0.09, text(raw"\(\sigma = 0.5\)", 11, BLUE, :left))

    return plt
end

plt = sigma_curve_plot()
savefig(plt, joinpath(@__DIR__, "sigma_curve.svg"))
display(plt)
