using Plots
pgfplotsx()

# colours consistent with bifurcation.jl
const GRAY = RGB(0.4, 0.4, 0.4)
const BLUE = RGB(0.15, 0.50, 0.82)

const XLO, XHI = -0.12, 1.28
const YLO, YHI = -0.12, 1.28

# s·t = κ, positive branch (κ = 0 is the axes)
function st_curve(κ; n = 400)
    κ == 0 && return ([0.0, 0.0, NaN, 0.0, XHI], [0.0, YHI, NaN, 0.0, 0.0])
    ss = range(κ / YHI, XHI, length = n)
    return ss, κ ./ ss
end

# The rendered axis is not square (aspect_ratio=1 is not honoured by pgfplotsx):
# it is AXIS_W × AXIS_H mm (read straight off sigma_curve.tikz's "width"/"height").
# Because the two x/y data ranges are equal, the x-direction is stretched by
# AXIS_W/AXIS_H on paper. The white box below is drawn in *data* coordinates, so it
# inherits this stretch and lines up with the curve automatically. But a pgfplots
# node `rotate=` acts in *on-paper* space, so a data-space tangent angle must be
# converted before it is handed to the text — otherwise the label sits 1–2° off the
# curve. (If `size` / margins / fonts change, re-read these two numbers.)
const AXIS_W, AXIS_H = 102.68, 95.06
const ASPECT = AXIS_H / AXIS_W                       # <1  ⇒  x stretched on paper
# data-space direction angle → on-paper angle pgfplots renders (and its inverse)
paper_angle(θ) = rad2deg(atan(ASPECT * sind(θ), cosd(θ)))
data_angle(θ)  = rad2deg(atan(sind(θ) / ASPECT, cosd(θ)))

function st_label!(plt, κ, label, t; rotation = nothing, fontsize = 9)
    s = κ == 0 ? 0.0 : κ / t
    if isnothing(rotation)
        θ = rad2deg(atan(-t / s))                    # data-space tangent (for the box)
        θ_text = paper_angle(θ)                      # on-paper angle for the label
    else
        θ_text = rotation                            # caller gives the on-paper angle
        θ = data_angle(rotation)                     # data-space angle for the box
    end
    u = (cosd(θ), sind(θ))
    v = (-sind(θ), cosd(θ))
    half_len, half_height = 0.125, 0.03
    xs = [
        s - half_len * u[1] - half_height * v[1],
        s + half_len * u[1] - half_height * v[1],
        s + half_len * u[1] + half_height * v[1],
        s - half_len * u[1] + half_height * v[1],
    ]
    ys = [
        t - half_len * u[2] - half_height * v[2],
        t + half_len * u[2] - half_height * v[2],
        t + half_len * u[2] + half_height * v[2],
        t - half_len * u[2] + half_height * v[2],
    ]
    plot!(plt, Shape(xs, ys); fillcolor = :white, linecolor = :white, label = "")
    annotate!(plt, s, t, text("\\(\\kappa = $label\\)", fontsize, GRAY, :center; rotation = θ_text))
end

# filled-triangle arrowhead, tip at (px, py) along (dx, dy)
function arrowhead!(plt, px, py, dx, dy; len = 0.06, col = :black)
    n = sqrt(dx^2 + dy^2)
    ux, uy = dx / n, dy / n                       # unit tangent
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

    # context curves
    for κ in [0.0, 0.08, 0.16]
        plot!(plt, st_curve(κ)...; color = GRAY, lw = 1, label = "")
    end
    st_label!(plt, 0.0, "0", 0.8; rotation = -90)
    st_label!(plt, 0.08, "0.1", 0.62)
    st_label!(plt, 0.16, "0.15", 0.45)
    # highlighted curve κ = κ0 along σ-parametrisation:
    #   s(σ) = √κ0 · exp(σ),  t(σ) = √κ0 · exp(-σ)
    κ0 = 0.25
    GREEN = RGB(0.18, 0.64, 0.30)
    σ1, σ2 = -0.5, 0.5
    scur(σ) = sqrt(κ0) * exp(σ)
    tcur(σ) = sqrt(κ0) * exp(-σ)
    σmin = log(sqrt(κ0) / YHI)                    # t at top frame
    σmax = log(XHI / sqrt(κ0))                    # s at right frame

    lerp(a, b, u) = a + (b - a) * u
    function σcolor(σ)
        if σ < σ1                                 # green side, fading out
            a = clamp((σ - σmin) / (σ1 - σmin), 0.0, 1.0)
            return RGBA(GREEN.r, GREEN.g, GREEN.b, a)
        elseif σ > σ2                             # blue side, fading out
            a = clamp((σmax - σ) / (σmax - σ2), 0.0, 1.0)
            return RGBA(BLUE.r, BLUE.g, BLUE.b, a)
        else                                      # green → blue between points
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

    # marked σ points
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