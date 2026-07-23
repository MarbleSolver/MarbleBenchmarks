using Plots
using LaTeXStrings
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
    σmid = (σ1 + σ2) / 2
    MIDCOL = σcolor(σmid)
    scatter!(plt, [scur(σmid)], [tcur(σmid)]; marker = :circle, ms = 7.5, color = MIDCOL,
        markerstrokecolor = :white, markerstrokewidth = 1.5, label = "")
    annotate!(plt, scur(σmid) + 0.04, tcur(σmid) + 0.09, text(L"(p_{\kappa}(\sigma),\, p_{\kappa}(-\sigma))", 11, MIDCOL, :left))

    # curved arrows tracing the contour toward the σ → ±∞ limits.
    # outward unit normal to the st = κ0 curve (∇(st) = (t, s), points away from origin)
    function outnorm(σ)
        s, t = scur(σ), tcur(σ)
        r = sqrt(s^2 + t^2)
        return (t / r, s / r)
    end
    # trace from σa to σb (arrowhead at σb), offset d outward, label at the arc midpoint.
    # the body stops at the arrowhead's base and the tip extends beyond, so they don't overlap.
    function contour_arrow!(σa, σb, col, lbl, σlbl; d = 0.05, n = 60, headlen = 0.03,
                            half_len = 0.105, fontsize = 9, pad_left = 0.0)
        opoint(σ) = (scur(σ) + d * outnorm(σ)[1], tcur(σ) + d * outnorm(σ)[2])
        xs, ys = Float64[], Float64[]
        for σ in range(σa, σb, length = n)
            p = opoint(σ); push!(xs, p[1]); push!(ys, p[2])
        end
        dx, dy = xs[end] - xs[end-1], ys[end] - ys[end-1]     # unit tangent at the far end
        len = sqrt(dx^2 + dy^2); ux, uy = dx / len, dy / len
        plot!(plt, xs, ys; color = col, lw = 1, label = "")               # body ends at base
        arrowhead!(plt, xs[end] + headlen * ux, ys[end] + headlen * uy, ux, uy;
            len = headlen, col = col)                                     # tip past the body
        # rotated label sitting on the arrow near its start, masking the line (like the κ labels)
        s, t = opoint(σlbl)
        θ = rad2deg(atan(-tcur(σlbl) / scur(σlbl)))           # data-space tangent
        θ_text = paper_angle(θ)                               # on-paper angle for the text
        u = (cosd(θ), sind(θ)); v = (-sind(θ), cosd(θ))
        half_height = 0.022
        lo = half_len + pad_left     # extra padding on the −u (σ) end of the box
        bx = [s - lo*u[1] - half_height*v[1], s + half_len*u[1] - half_height*v[1],
              s + half_len*u[1] + half_height*v[1], s - lo*u[1] + half_height*v[1]]
        by = [t - lo*u[2] - half_height*v[2], t + half_len*u[2] - half_height*v[2],
              t + half_len*u[2] + half_height*v[2], t - lo*u[2] + half_height*v[2]]
        plot!(plt, Shape(bx, by); fillcolor = :white, linecolor = :white, label = "")
        annotate!(plt, s, t, text(lbl, fontsize, col, :center; rotation = θ_text))
    end
    contour_arrow!(-0.30, -0.80, GREEN, L"\sigma < 0", -0.42)
    contour_arrow!(0.30, 0.80, BLUE, L"\sigma > 0", 0.42)

    return plt
end

# Pad the saved SVG (non-distorting: transparent margin, content stays 1:1) to a
# fixed canvas so sigma_curve.svg and solnchange.svg come out at identical size.
# Keep CANVAS_PT in sync with solnchange.jl.
const CANVAS_PT = (294.0, 264.0)
function pad_svg_canvas!(path, W, H)
    svg = read(path, String)
    m = match(r"width=\"([0-9.]+)pt\" height=\"([0-9.]+)pt\" viewBox=\"0 0 ([0-9.]+) ([0-9.]+)\"", svg)
    m === nothing && error("could not parse SVG size header in $path")
    w, h = parse(Float64, m[1]), parse(Float64, m[2])
    (w > W + 1e-6 || h > H + 1e-6) &&
        error("$(basename(path)) is $(w)×$(h)pt > target $(W)×$(H)pt; bump CANVAS_PT in both scripts")
    px, py = (W - w) / 2, (H - h) / 2
    hdr = "width=\"$(W)pt\" height=\"$(H)pt\" viewBox=\"$(-px) $(-py) $(W) $(H)\""
    write(path, replace(svg, m.match => hdr; count = 1))
end

# also emit a PDF at the same canvas size, converted from the padded SVG (so the
# PDF and SVG — and the sigma_curve/solnchange pair — share identical dimensions).
function svg_to_pdf(svgpath)
    pdfpath = replace(svgpath, r"\.svg$" => ".pdf")
    try
        run(`inkscape $svgpath --export-type=pdf --export-filename=$pdfpath`)
    catch err
        @warn "PDF export via Inkscape failed (is it on PATH?); SVG still written" err
    end
end

plt = sigma_curve_plot()
out = joinpath(@__DIR__, "sigma_curve.svg")
savefig(plt, out)
pad_svg_canvas!(out, CANVAS_PT...)
svg_to_pdf(out)
display(plt)