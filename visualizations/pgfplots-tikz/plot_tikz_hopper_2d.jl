using PGFPlotsX, MarbleBenchmarks, Serialization, LinearAlgebra

# ===========================================================================
# 2-D hopper trajectory — side (x–z) view (PGFPlotsX / TikZ).
#
#   * the staircase terrain (three rising steps, then a drop-off at x = 2 m) is
#     drawn as a filled gray profile.
#   * the hopper is a two-mass leg (foot + head): at every knot the leg is drawn
#     faintly as a "strobe", building up the motion; the foot and head paths are
#     overlaid as coloured trajectory lines with a dot at every knot.
#   * knots where the contact normal force is nonzero (the foot is in stance) are
#     highlighted: the stance leg is drawn solid in the contact colour and the
#     contact force vector is drawn as an arrow from the foot, scaled by |f|.
# ===========================================================================

# --- load the Marble solution ----------------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

# allow overriding the solution file on the command line, else take the latest
solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

prob = get_problem(:hopper_2d)
inds, p = prob.inds, prob.params
N = p.N

z = deserialize(solfile)["hopper_2d"]
X = z[inds[:x]]        # (N+1) × 8  state [foot_x, foot_z, head_x, head_z, ...]
F = z[inds[:f]]        # N × 2      contact force [tangential; normal]

# The model keeps the two masses nearly coincident (a short prismatic leg), so we
# reuse the animation's visual offsets to draw a recognisable hopper leg: the foot
# sphere sits at r above its contact point, the head sits h + r higher still.
r, h = 0.05, 0.25
foot_x, foot_z = X[:, 1], X[:, 2]
head_x, head_z = X[:, 3], X[:, 4] .+ (h + r)

# a knot is a stance knot when the contact normal force is nonzero (up to noise)
const F_TOL = 2e-5
stance = [k for k in 1:N if F[k, 2] > F_TOL]
stance_set = Set(stance)
flight = [k for k in 1:N+1 if !(k in stance_set)]   # all inactive (flight) knots
fmag_max = maximum(norm(F[k, :]) for k in 1:N)
const FARROW = 0.30 / fmag_max      # plot-units per unit force

ff(v) = string(round(v, digits = 5))

# staircase terrain profile (x, z), left to right, inferred from the model /
# reference geometry: floor, three 0.1 m steps, then a drop-off back to 0 at x = 2.
const XL, XR, ZB = -0.15, 3.35, -0.18
terrain = [(XL, 0.0), (0.5, 0.0), (0.5, 0.1), (0.75, 0.1), (0.75, 0.2),
           (1.0, 0.2), (1.0, 0.3), (2.0, 0.3), (2.0, 0.0), (XR, 0.0)]

# ===========================================================================
# preamble: colours + styles
# ===========================================================================
empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usetikzlibrary{arrows.meta}
\definecolor{ctraj}{HTML}{2B2F36}     % foot trajectory / hopper body (near-black slate)
\definecolor{cknot}{HTML}{6E747C}     % flight-phase leg strobe
\definecolor{ccont}{HTML}{CB4154}     % contact-force arrow (strong crimson)
\definecolor{cbody}{HTML}{E39AA0}     % stance body (light crimson)
\definecolor{cterr}{HTML}{555A61}     % terrain edge
\pgfplotsset{
  compat=1.18,
  hview/.style={
    axis equal image,
    axis line style={gray!60, line width=0.5pt},
    tick style={gray!60, line width=0.4pt},
    tick align=outside,
    tick label style={font=\footnotesize},
    label style={font=\small},
    title style={font=\small\bfseries, yshift=3pt},
    grid=major,
    major grid style={gray!14, line width=0.3pt},
    line width=0.9pt,
    clip=false,
  },
}
\tikzset{
  legflight/.style={draw=cknot, line width=0.7pt, line cap=round},
  legstance/.style={draw=cbody, line width=2.0pt, line cap=round},
  forcearrow/.style={draw=ccont, line width=1.3pt, -{Latex[length=4.5pt]}},
  terredge/.style={draw=cterr, line width=1.1pt, line join=round},
  annot/.style={font=\footnotesize},
}
""")

# terrain: filled profile closed down to the baseline, plus a darker top edge ---
terr_path = join(["(axis cs:$(ff(x)),$(ff(zz)))" for (x, zz) in terrain], " -- ")
terrain_fill = "\\fill[cterr!12] (axis cs:$(ff(XL)),$(ff(ZB))) -- " * terr_path *
               " -- (axis cs:$(ff(XR)),$(ff(ZB))) -- cycle;"
terrain_edge = "\\draw[terredge] " * terr_path * ";"

# leg strobe: a faint segment foot -> head at every flight knot ----------------
leg_draws = String[]
for k in 1:N+1
    k in stance && continue
    push!(leg_draws,
        "\\draw[legflight] (axis cs:$(ff(foot_x[k])),$(ff(foot_z[k]))) -- " *
        "(axis cs:$(ff(head_x[k])),$(ff(head_z[k])));")
end

# stance: bold leg + force-vector arrow at every contact knot -------------------
stance_draws = String[]
for k in stance
    push!(stance_draws,
        "\\draw[legstance] (axis cs:$(ff(foot_x[k])),$(ff(foot_z[k]))) -- " *
        "(axis cs:$(ff(head_x[k])),$(ff(head_z[k])));")
    fx = foot_x[k] + F[k, 1] * FARROW
    fz = foot_z[k] + F[k, 2] * FARROW
    push!(stance_draws,
        "\\draw[forcearrow] (axis cs:$(ff(foot_x[k])),$(ff(foot_z[k]))) -- " *
        "(axis cs:$(ff(fx)),$(ff(fz)));")
end
stance_fx = [foot_x[k] for k in stance]
stance_fz = [foot_z[k] for k in stance]

# compact legend in the empty top-right region ---------------------------------
lx, lz, dz = 2.45, 0.74, 0.085
legend = [
    "\\draw[ctraj, line width=1.2pt] (axis cs:$(ff(lx)),$(ff(lz))) -- (axis cs:$(ff(lx+0.16)),$(ff(lz)));",
    "\\node[annot, anchor=west] at (axis cs:$(ff(lx+0.20)),$(ff(lz))) {foot path};",
    "\\draw[forcearrow] (axis cs:$(ff(lx)),$(ff(lz-dz))) -- (axis cs:$(ff(lx+0.16)),$(ff(lz-dz)));",
    "\\node[annot, anchor=west] at (axis cs:$(ff(lx+0.20)),$(ff(lz-dz))) {contact force};",
]

# ===========================================================================
# assemble the plot
# ===========================================================================
ax = @pgf Axis(
    {
        "hview", width = "17cm",
        xlabel = raw"$x$ $[\mathrm{m}]$",
        ylabel = raw"$z$ $[\mathrm{m}]$",
        xmin = XL, xmax = XR, ymin = ZB, ymax = 0.8,
        xtick = "{0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2}",
    },
    terrain_fill,
    terrain_edge,
    leg_draws...,
    # stance (contact) legs + force arrows
    stance_draws...,
    # foot path
    @pgf(Plot({no_marks, color = "ctraj", line_width = "1.2pt"},
              Coordinates(foot_x, foot_z))),
    # inactive (flight) hoppers: feet + heads in light gray
    @pgf(Plot({only_marks, mark = "*", mark_size = "1.1pt",
               mark_options = "{fill=cknot, draw=cknot}"}, Coordinates(foot_x[flight], foot_z[flight]))),
    # active (stance) hoppers: feet in faded red
    @pgf(Plot({only_marks, mark = "*", mark_size = "1.1pt",
               mark_options = "{fill=cbody, draw=cbody}"}, Coordinates(foot_x[stance], foot_z[stance]))),
    # heads drawn last (on top), larger than feet, coloured by phase
    @pgf(Plot({only_marks, mark = "*", mark_size = "2.1pt",
               mark_options = "{fill=cknot, draw=cknot}"}, Coordinates(head_x[flight], head_z[flight]))),
    @pgf(Plot({only_marks, mark = "*", mark_size = "2.1pt",
               mark_options = "{fill=cbody, draw=cbody}"}, Coordinates(head_x[stance], head_z[stance]))),
    legend...,
)

fig = @pgf TikzPicture(ax)

outsvg = joinpath(@__DIR__, "hopper_2d_sideview.svg")
pgfsave(outsvg, fig)
pgfsave(joinpath(@__DIR__, "hopper_2d_sideview.pdf"), fig)
@info "wrote $outsvg  ($(length(stance)) stance knots: $(stance))"
