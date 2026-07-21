using PGFPlotsX, MarbleBenchmarks, Serialization, LinearAlgebra

# ===========================================================================
# Rocket-catch trajectory — clean orthographic three-view (PGFPlotsX / TikZ).
#
#   (a) lateral side view  x–z   ┐  both side views highlight the
#   (b) approach side view y–z   ┘  trigger altitude  z = z_trig
#   (c) top / plan view    x–y      highlights the trapezoidal keep-in region
#
# Frame:  x = lateral,  y = range to the tower (tower face at y = 0),
#         z = altitude.  Units are km (state is scaled Super-Heavy-ish).
# ===========================================================================

# --- geometry helpers, mirrored from src/problems/rocket_catch_3d.jl --------
Rx(θ) = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
Ry(θ) = [cos(θ) 0 sin(θ); 0 1.0 0; -sin(θ) 0 cos(θ)]
hatm(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
Lq(q) = (s = q[1]; v = q[2:4]; [s -v'; v s*I(3)+hatm(v)])
const _Tq = Diagonal([1.0; -ones(3)]); const _Hq = [zeros(1, 3); I(3)]
qtoQ(q) = _Hq' * _Tq * Lq(q) * _Tq * Lq(q) * _Hq
rptoq(ϕ) = (1 / sqrt(1 + ϕ'ϕ)) * [1.0; ϕ]
thrust_axis_body(δr, δp) = Rx(δr) * Ry(δp) * [0.0, 0.0, 1.0]

# --- load the Marble solution -----------------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

prob = get_problem(:rocket_catch_3d)
inds, p = prob.inds, prob.params
X = deserialize(latest_marble())["rocket_catch_3d"][inds[:x]]     # N × 14
N = p.N
xs, ys, zs = X[:, 1], X[:, 2], X[:, 3]

zt, zc = p.trigger_altitude, p.catch_altitude
hw, slope = p.arm_half_width, tan(p.arm_opening_angle)

# body long axis (engine->nose) and thrust-force direction, world frame, per knot
baxis = [qtoQ(rptoq(X[k, 4:6])) * [0.0, 0.0, 1.0] for k in 1:N]
tdir  = [qtoQ(rptoq(X[k, 4:6])) * thrust_axis_body(X[k, 13], X[k, 14]) for k in 1:N]

# --- split the path at the exact trigger-altitude crossing ------------------
kc = findfirst(<=(zt), zs)                       # first knot below the trigger
frac = (zs[kc-1] - zt) / (zs[kc-1] - zs[kc])
xstar = xs[kc-1] + frac * (xs[kc] - xs[kc-1])
ystar = ys[kc-1] + frac * (ys[kc] - ys[kc-1])

above(v, vstar) = [v[1:kc-1]; vstar]             # above trigger (coasting)
below(v, vstar) = [vstar; v[kc:N]]               # below trigger (constrained)
xa, xb = above(xs, xstar), below(xs, xstar)      # full (used by the top view)
ya, yb = above(ys, ystar), below(ys, ystar)
za, zb = above(zs, zt),   below(zs, zt)

# --- drawing extents (altitude cropped: the high-altitude coast is not shown)
xlo, xhi = -0.09, 0.09
ylo, yhi = -0.014, 0.135
zlo, zhi = 0.06, 0.35
ff(v) = string(round(v, digits = 5))             # coordinate formatter

# coasting arc, cropped to the visible altitude window for the SIDE views. The
# arc enters cleanly at the top border (interpolated z = zhi) instead of running
# off the plot.
k35 = findfirst(<=(zhi), zs)
f35 = (zs[k35-1] - zhi) / (zs[k35-1] - zs[k35])
x35 = xs[k35-1] + f35 * (xs[k35] - xs[k35-1])
y35 = ys[k35-1] + f35 * (ys[k35] - ys[k35-1])
coast = k35:(kc-1)
xac = [x35; xs[coast]; xstar]; yac = [y35; ys[coast]; ystar]; zac = [zhi; zs[coast]; zt]

# ===========================================================================
# preamble: colours + one shared axis style + annotation styles
# ===========================================================================
empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usetikzlibrary{arrows.meta,patterns,patterns.meta,angles,quotes}
\usepgfplotslibrary{fillbetween}
\definecolor{cabove}{HTML}{2C5F8A}   % coasting, above trigger
\definecolor{cbelow}{HTML}{D9552B}   % constrained, below trigger
\definecolor{cfun}{HTML}{2C5F8A}     % keep-in region
\definecolor{ctrig}{HTML}{B3352B}    % trigger altitude
\definecolor{ctower}{HTML}{555A61}   % tower / arms
\definecolor{cbody}{HTML}{6E747C}    % rocket body
\pgfplotsset{
  compat=1.18,
  rcview/.style={
    scale only axis,
    enlargelimits=false,
    scaled ticks=false,
    axis line style={gray!60, line width=0.5pt},
    tick style={gray!60, line width=0.4pt},
    tick align=outside,
    tick label style={font=\footnotesize, /pgf/number format/fixed, /pgf/number format/precision=2},
    label style={font=\small},
    title style={font=\small\bfseries, yshift=3pt},
    every axis title/.append style={align=center},
    grid=major,
    major grid style={gray!14, line width=0.3pt},
    line width=0.9pt,
    clip=false,
  },
}
\tikzset{
  trigline/.style={ctrig, dashed, dash pattern=on 3pt off 2pt, line width=0.8pt},
  rocketbody/.style={draw=cbody, line width=1.9pt, line cap=round},
  gimbal/.style={draw=cbelow, line width=1.0pt, line cap=round},
  annot/.style={font=\footnotesize},
}
""")

# common per-axis size and style ------------------------------------------------
Wx, Wy, Hz, Hy = "5.4cm", "4.0cm", "6.0cm", "4.8cm"
lw_traj = "1.2pt"

# knots at which to draw a rocket glyph (spread over the visible altitudes)
gsel = unique(argmin(abs.(zs .- z)) for z in (0.315, 0.245, 0.180, 0.128, 0.098))
Lbody = 0.028     # drawn body length (schematic attitude indicator)
Lgimb = 0.011     # gimbal / nozzle stub length

# rocket glyph in a projection: `h`,`v` pick which world components map to the
# horizontal / vertical plot axes (1=x, 2=y, 3=z). Gray body along the attitude
# axis, plus a short stub at the engine along the (gimballed) thrust axis.
function glyphs(h, v)
    raws = String[]
    for k in gsel
        b, t = baxis[k], tdir[k]
        cx, cv = X[k, h], X[k, v]
        bsx, bsv = cx - 0.5Lbody * b[h], cv - 0.5Lbody * b[v]   # engine
        tpx, tpv = cx + 0.5Lbody * b[h], cv + 0.5Lbody * b[v]   # nose
        gx, gv = bsx - Lgimb * t[h], bsv - Lgimb * t[v]         # nozzle stub
        push!(raws, "\\draw[rocketbody] (axis cs:$(ff(bsx)),$(ff(bsv))) -- (axis cs:$(ff(tpx)),$(ff(tpv)));")
        push!(raws, "\\draw[gimbal] (axis cs:$(ff(bsx)),$(ff(bsv))) -- (axis cs:$(ff(gx)),$(ff(gv)));")
    end
    raws
end

traj_above(cx, cy) = @pgf Plot({no_marks, color = "cabove", line_width = lw_traj},
                               Coordinates(cx, cy))
traj_below(cx, cy) = @pgf Plot({no_marks, color = "cbelow", line_width = lw_traj},
                               Coordinates(cx, cy))
knot_dots(cx, cy, col) = @pgf Plot({only_marks, mark = "*", mark_size = "0.7pt",
                                    mark_options = "{fill=$col, draw=$col}"},
                                   Coordinates(cx, cy))
start_mark(cx, cy) = @pgf Plot({only_marks, mark = "o", mark_size = "2.6pt",
                                color = "cabove", line_width = "0.9pt"},
                               Coordinates([cx], [cy]))
catch_mark(cx, cy) = @pgf Plot({only_marks, mark = "star", mark_size = "3.4pt",
                                color = "cbelow", line_width = "0.9pt"},
                               Coordinates([cx], [cy]))

# ---------------------------------------------------------------------------
# (a) approach side view : y (horizontal) vs z (vertical)
# ---------------------------------------------------------------------------
side_y = @pgf Axis(
    {
        "rcview", name = "sidey", width = Wy, height = Hz,
        xmin = ylo, xmax = yhi, ymin = zlo, ymax = zhi,
        xtick = "{0,0.05,0.1}", ytick = "{0.1,0.15,0.2,0.25,0.3,0.35}",
        xlabel = raw"$y$, range to tower $[\mathrm{km}]$",
        ylabel = raw"$z$, altitude $[\mathrm{km}]$",
        title = raw"(a) approach side view",
    },
    raw"\fill[ctrig, opacity=0.05] (axis cs:-0.014,0.06) rectangle (axis cs:0.135,0.12);",
    raw"\draw[trigline] (axis cs:-0.014,0.12) -- (axis cs:0.135,0.12);",
    raw"\node[annot, ctrig, anchor=south east] at (axis cs:0.133,0.121) {$z_{\mathrm{trig}}$};",
    raw"\draw[ctower, dotted, line width=0.7pt] (axis cs:-0.014,0.08) -- (axis cs:0.135,0.08);",
    # tower face at y = 0 (rocket must stay at y >= 0)
    raw"\fill[pattern={Lines[angle=45,distance=2.4pt,line width=0.4pt]}, pattern color=ctower!70] (axis cs:-0.014,0.06) rectangle (axis cs:0,0.30);",
    raw"\draw[ctower, line width=0.8pt] (axis cs:0,0.06) -- (axis cs:0,0.30);",
    traj_above(yac, zac), traj_below(yb, zb),
    glyphs(2, 3)...,
    catch_mark(ys[N], zs[N]),
)

# ---------------------------------------------------------------------------
# (b) top / plan view : x (horizontal) vs y (vertical)
# ---------------------------------------------------------------------------
# zoomed extents for the top view
txmin, txmax, tymin, tymax = -0.055, 0.088, -0.011, 0.125
xg = hw + slope * tymax       # keep-in half-width at the (zoomed) far edge
# left-face angle annotation: vertex, vertical reference, point on the face
yang = 0.055
top = @pgf Axis(
    {
        "rcview", name = "top", width = Wx, height = Hy,
        at = "(sidey.south east)", anchor = "south west", xshift = "1.3cm",
        xmin = txmin, xmax = txmax, ymin = tymin, ymax = tymax,
        xtick = "{-0.05,0,0.05}", ytick = "{0,0.05,0.1}",
        xlabel = raw"$x$, lateral $[\mathrm{km}]$",
        ylabel = raw"$y$, range $[\mathrm{km}]$",
        title = raw"(b) top view",
    },
    # trapezoidal keep-in region  |x| <= hw + slope*y ,  y >= 0
    "\\fill[cfun, opacity=0.10] (axis cs:$(ff(-hw)),0) -- (axis cs:$(ff(hw)),0) -- " *
        "(axis cs:$(ff(xg)),$(ff(tymax))) -- (axis cs:$(ff(-xg)),$(ff(tymax))) -- cycle;",
    "\\draw[cfun, line width=0.8pt] (axis cs:$(ff(hw)),0) -- (axis cs:$(ff(xg)),$(ff(tymax)));",
    "\\draw[cfun, line width=0.8pt] (axis cs:$(ff(-hw)),0) -- (axis cs:$(ff(-xg)),$(ff(tymax)));",
    raw"\node[annot, cfun!85, anchor=south] at (axis cs:0.028,0.108) {keep-in region};",
    # tower arms at y = 0 with the catch gap between them
    "\\fill[pattern={Lines[angle=45,distance=2.4pt,line width=0.4pt]}, pattern color=ctower!70] (axis cs:$(ff(txmin)),$(ff(tymin))) rectangle (axis cs:$(ff(-hw)),0);",
    "\\fill[pattern={Lines[angle=45,distance=2.4pt,line width=0.4pt]}, pattern color=ctower!70] (axis cs:$(ff(hw)),$(ff(tymin))) rectangle (axis cs:$(ff(txmax)),0);",
    "\\draw[ctower, line width=0.8pt] (axis cs:$(ff(txmin)),0) -- (axis cs:$(ff(-hw)),0);",
    "\\draw[ctower, line width=0.8pt] (axis cs:$(ff(hw)),0) -- (axis cs:$(ff(txmax)),0);",
    # arm opening half-angle theta_arm, between the left face and the vertical
    "\\coordinate (Oarm) at (axis cs:$(ff(-hw)),0);",
    "\\coordinate (Varm) at (axis cs:$(ff(-hw)),$(ff(yang)));",
    "\\coordinate (Farm) at (axis cs:$(ff(-hw-slope*yang)),$(ff(yang)));",
    raw"\draw[ctower, dashed, line width=0.5pt] (Oarm) -- (Varm);",
    "\\pic[draw=black, line width=0.5pt, angle radius=7mm, angle eccentricity=1.55, \"\$\\theta_{\\mathrm{arm}}\$\"{font=\\footnotesize}]{angle=Varm--Oarm--Farm};",
    # trajectory projection
    traj_above(xa, ya), traj_below(xb, yb),
    start_mark(xs[1], ys[1]), catch_mark(xs[N], ys[N]),
)

# ---------------------------------------------------------------------------
# shared legend, placed to the right of the top row
# ---------------------------------------------------------------------------
legend = @pgf Axis(
    {
        name = "leg", at = "(top.north east)", anchor = "north west",
        xshift = "0.8cm",
        hide_axis, scale_only_axis, width = "0.6cm", height = Hy,
        xmin = 0, xmax = 1, ymin = 0, ymax = 1, clip = false,
    },
    raw"\draw[cabove, line width=1.2pt] (axis cs:0,0.96) -- (axis cs:1,0.96); \node[annot, anchor=west] at (axis cs:1.25,0.96) {trajectory (coasting)};",
    raw"\draw[cbelow, line width=1.2pt] (axis cs:0,0.895) -- (axis cs:1,0.895); \node[annot, anchor=west] at (axis cs:1.25,0.895) {trajectory (below $z_{\mathrm{trig}}$)};",
    raw"\draw[trigline] (axis cs:0,0.83) -- (axis cs:1,0.83); \node[annot, anchor=west] at (axis cs:1.25,0.83) {trigger altitude $z_{\mathrm{trig}}$};",
    raw"\draw[rocketbody] (axis cs:0.15,0.765) -- (axis cs:0.85,0.765); \node[annot, anchor=west] at (axis cs:1.25,0.765) {rocket body (attitude)};",
    raw"\draw[gimbal] (axis cs:0.3,0.70) -- (axis cs:0.7,0.685); \node[annot, anchor=west] at (axis cs:1.25,0.70) {engine gimbal};",
    raw"\fill[cfun, opacity=0.18] (axis cs:0,0.60) rectangle (axis cs:1,0.66); \draw[cfun, line width=0.6pt] (axis cs:0,0.60) rectangle (axis cs:1,0.66); \node[annot, anchor=west] at (axis cs:1.25,0.63) {keep-in region};",
    raw"\fill[pattern={Lines[angle=45,distance=2.4pt,line width=0.4pt]}, pattern color=ctower!70] (axis cs:0,0.52) rectangle (axis cs:1,0.58); \draw[ctower] (axis cs:0,0.52) rectangle (axis cs:1,0.58); \node[annot, anchor=west] at (axis cs:1.25,0.55) {tower / arms};",
    raw"\node[cabove] at (axis cs:0.5,0.475) {$\circ$}; \node[annot, anchor=west] at (axis cs:1.25,0.475) {entry point};",
    raw"\node[cbelow] at (axis cs:0.5,0.41) {$\star$}; \node[annot, anchor=west] at (axis cs:1.25,0.41) {catch point};",
)

fig = @pgf TikzPicture(side_y, top, legend)

outfile = joinpath(@__DIR__, "rocket_catch_3d.svg")
pgfsave(outfile, fig)
pgfsave(joinpath(@__DIR__, "rocket_catch_3d.pdf"), fig)
@info "wrote $outfile"
