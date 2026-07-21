using PGFPlotsX

# ===========================================================================
# Standalone start/end legend (PGFPlotsX / TikZ) — a single row
#
#     ● Start    ● End
#
# meant to be dropped under the GLMakie renders in png_renders/, which colour
# a trajectory by time. The two dot colours are the *endpoints* of the
# :diverging_rainbow_bgymr_45_85_c67_n256 gradient those scripts use, so the
# blue dot is exactly the colour of the first knot and the red dot the last.
# There is no axis and no frame: it is just the key.
# ===========================================================================

const START_HEX = "085CF8"   # cgrad(:diverging_rainbow_bgymr_45_85_c67_n256)[1]
const END_HEX   = "D70500"   # cgrad(:diverging_rainbow_bgymr_45_85_c67_n256)[end]

empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usetikzlibrary{positioning}
\definecolor{cstart}{HTML}{""" * START_HEX * raw"""}
\definecolor{cend}{HTML}{""" * END_HEX * raw"""}
\tikzset{
  % white hairline around each dot, as on the highlighted knots in the
  % pgfplots_tikz/ trajectory plots, so the marker reads on any background
  legdot/.style={circle, minimum size=5.5pt, inner sep=0pt,
                 draw=white, line width=0.4pt},
  leglabel/.style={font=\small, text=black, inner sep=0pt},
}
""")

# nodes are chained with `right=of`, so the spacing holds whatever the labels say
fig = @pgf TikzPicture(
    raw"\node[legdot, fill=cstart] (sd) at (0,0) {};",
    raw"\node[leglabel, right=4pt of sd] (sl) {Start};",
    raw"\node[legdot, fill=cend, right=14pt of sl] (ed) {};",
    raw"\node[leglabel, right=4pt of ed] (el) {End};",
)

outsvg = joinpath(@__DIR__, "start_end_legend.svg")
pgfsave(outsvg, fig)
pgfsave(joinpath(@__DIR__, "start_end_legend.pdf"), fig)
@info "wrote $outsvg"
