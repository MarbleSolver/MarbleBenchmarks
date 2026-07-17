using PGFPlotsX, MarbleBenchmarks, Serialization, LinearAlgebra

# ===========================================================================
# Quadrotor-gates trajectory — top-down (x–y) plan view (PGFPlotsX / TikZ).
#
#   * each gate is drawn as a line segment (its width axis), one bright colour
#     per gate; a small legend "1 -> 2 -> 3 -> 4" at the top right maps the
#     colours to the passing order.
#   * the flown quadrotor trajectory is drawn with a dot at every knot.
#   * knots where a progress variable μ is nonzero (a "progress knot", i.e. the
#     gate is registered as passed there) are marked simply by recolouring that
#     knot's dot in the corresponding gate's colour — no extra glyph.
# ===========================================================================

# --- load the Marble solution ----------------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

# allow overriding the solution file on the command line, else take the latest
solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

prob = get_problem(:quadrotor_gates_3d)
inds, p = prob.inds, prob.params
N, ngates = p.N, p.ngates

z = deserialize(solfile)["quadrotor_gates_3d"]
X = z[inds[:x]]        # N × 9   (Δ-state; x̄[1:3] = 0 so cols 1:2 are world x,y)
μ = z[inds[:μ]]        # N × ngates   progress variables

xs = p.x̄[1] .+ X[:, 1]
ys = p.x̄[2] .+ X[:, 2]

# a knot is a "progress knot" for gate j when μ[k,j] is nonzero (up to noise)
const MU_TOL = 1e-3
progress = [(k, argmax(μ[k, :])) for k in 1:N if maximum(μ[k, :]) > MU_TOL]

# gate line segments in the ground plane: the gate width axis is the gate's
# local-x (first column of Rz), spanning ±h1 about the gate centre.
h1 = p.gate_dims[1] / 2
gate_segments = map(1:ngates) do j
    c = p.gate_positions[j][1:2]
    w = p.gate_orientations[j][1:2, 1]        # width direction in the xy-plane
    (c .- h1 .* w, c .+ h1 .* w)
end
ff(v) = string(round(v, digits = 5))

# ===========================================================================
# preamble: colours + styles
# ===========================================================================
# one bright, distinct colour per gate (blue / gold / green / crimson) — a
# qualitative palette in the spirit of the sigma-curve / solnchange plots.
const GATE_HEX = ["2C6FB0", "E6A817", "2E9E52", "CB4154"]

empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usetikzlibrary{arrows.meta}
\definecolor{ctraj}{HTML}{2B2F36}    % trajectory (near-black slate)
\definecolor{cknot}{HTML}{6E747C}    % highlighted γ knots
\definecolor{cends}{HTML}{2B2F36}    % start / goal text
""" *
    join(["\\definecolor{g$j}{HTML}{$(GATE_HEX[j])}" for j in 1:ngates], "\n") *
raw"""
\pgfplotsset{
  compat=1.18,
  qview/.style={
    scale only axis,
    axis equal,
    enlargelimits=0.12,
    axis line style={gray!60, line width=0.5pt},
    tick style={gray!60, line width=0.4pt},
    tick align=outside,
    tick label style={font=\footnotesize},
    label style={font=\small},
    grid=major,
    major grid style={gray!14, line width=0.3pt},
    line width=0.9pt,
    clip=false,
  },
}
\tikzset{
  annot/.style={font=\footnotesize},
}
""")

# gate segments, one colour per gate (no text labels) -----------------------
gate_draws = String[]
for (j, (a, b)) in enumerate(gate_segments)
    push!(gate_draws,
        "\\draw[draw=g$j, line width=2.6pt, line cap=round] " *
        "(axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end

# ordering legend "1 -> 2 -> 3 -> 4", each index in its gate colour ----------
legend_items = ["\\textbf{\\textcolor{g$j}{$j}}" for j in 1:ngates]
legend_str = "\\node[anchor=north east, draw=gray!40, fill=white, rounded corners=2pt, " *
    "inner sep=4pt, line width=0.5pt, font=\\small] at (rel axis cs:0.985,0.985) {" *
    join(legend_items, "\\,\$\\to\$\\,") * "};"

# progress knots: recolour the knot dot in the passed gate's colour, with a
# thin white border to lift it off the trajectory.
prog_plots = [
    @pgf Plot({only_marks, mark = "*", mark_size = "1.9pt",
               mark_options = "{fill=g$j, draw=white, line width=0.5pt}"},
              Coordinates([xs[k]], [ys[k]]))
    for (k, j) in progress
]

# annotate each progress knot with μ^(j) = 1 in the gate's colour. slight
# upward tilt + a mostly-opaque white backing so the label stays legible where
# it crosses the trajectory.
prog_labels = [
    "\\node[annot, text=g$j, rotate=18, anchor=south west, inner sep=1.5pt, " *
    "fill=white, fill opacity=0.85, text opacity=1, rounded corners=1pt] " *
    "at (axis cs:$(ff(xs[k])),$(ff(ys[k]))) {\$\\mu^{($j)} = 1\$};"
    for (k, j) in progress
]

# three highlighted γ (cumulative-progress) knots: start, one between gates 3
# and 4 (closer to gate 4), and the goal — illustrating how γ ratchets up.
kmid = 45
gamma_knots = [
    (1,    "\\gamma = (0,0,0,0)", "east"),   # start  — label to the left
    (kmid, "\\gamma = (1,1,1,0)", "north"),  # mid    — label below
    (N,    "\\gamma = (1,1,1,1)", "west"),   # goal   — label to the right
]
gamma_ks = [k for (k, _, _) in gamma_knots]
gamma_labels = [
    "\\node[annot, cends, anchor=$anch, inner sep=4pt, fill=white, " *
    "fill opacity=0.85, text opacity=1, rounded corners=1pt] " *
    "at (axis cs:$(ff(xs[k])),$(ff(ys[k]))) {\$$lbl\$};"
    for (k, lbl, anch) in gamma_knots
]

# ===========================================================================
# assemble the plot
# ===========================================================================
ax = @pgf Axis(
    {
        "qview", width = "9cm", height = "9cm",
        xlabel = raw"$x$ $[\mathrm{m}]$",
        ylabel = raw"$y$ $[\mathrm{m}]$",
    },
    gate_draws...,
    # trajectory line
    @pgf(Plot({no_marks, color = "ctraj", line_width = "1.2pt"}, Coordinates(xs, ys))),
    # three highlighted γ knots (start / mid / goal)
    @pgf(Plot({only_marks, mark = "*", mark_size = "2.4pt",
               mark_options = "{fill=cknot, draw=white, line width=0.5pt}"},
              Coordinates(xs[gamma_ks], ys[gamma_ks]))),
    gamma_labels...,
    # progress knots recoloured by gate + μ label
    prog_plots...,
    prog_labels...,
    # start / goal text
    raw"\node[annot, cends, anchor=north, yshift=-2pt] at (axis cs:" * ff(xs[1]) * "," * ff(ys[1]) * ") {start};",
    raw"\node[annot, cends, anchor=south, yshift=2pt] at (axis cs:" * ff(xs[N]) * "," * ff(ys[N]) * ") {goal};",
    legend_str,
)

fig = @pgf TikzPicture(ax)

outsvg = joinpath(@__DIR__, "quadrotor_gates_3d_topdown.svg")
pgfsave(outsvg, fig)
pgfsave(joinpath(@__DIR__, "quadrotor_gates_3d_topdown.pdf"), fig)
@info "wrote $outsvg  ($(length(progress)) progress knots: $(progress))"
