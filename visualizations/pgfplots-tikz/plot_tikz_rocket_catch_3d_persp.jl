using PGFPlotsX, MarbleBenchmarks, Serialization, LinearAlgebra

# attitude helpers (reference quaternion is identity, so q = rptoq(δrp) and the
# body→world rotation is qtoQ(q)).
const rptoq = MarbleBenchmarks.rptoq
const qtoQ  = MarbleBenchmarks.qtoQ
const rocket_thrust_axis_body = MarbleBenchmarks.rocket_thrust_axis_body

# ===========================================================================
# Rocket-catch (Mechazilla) trajectory — 3-D PERSPECTIVE view (PGFPlotsX /
# TikZ), rendered in the same style as plot_tikz_quadrotor_gates_3d_persp.jl.
#
# Every world point (trajectory knots, funnel corners, tower, rocket bodies,
# ground grid) is passed through a look-at + perspective pinhole camera to a
# 2-D screen coordinate, then emitted into a hidden-axis TikZ plot.
#
# Scene elements
#   • depth-shaded (near-black → light-gray) descent trajectory of the CoM
#   • gray tower body with two flat trapezoidal chopstick catch arms
#   • thin gray rocket cylinders at a few knots, each at its solved attitude with
#     a thin red engine-gimbal line out of the base
#   • the trapezoidal catch "funnel" corridor (|x| <= aw + slope·y, y >= 0),
#     extruded up to the trigger altitude, whose translucent top face marks
#     z_trig.
# ===========================================================================

# --- load the Marble solution ----------------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

prob = get_problem(:rocket_catch_3d)
inds, p = prob.inds, prob.params
N = p.N

z = deserialize(solfile)["rocket_catch_3d"]
X = z[inds[:x]]        # N × 14  reduced state [p(3); ϕ(3); v(3); ω(3); δr; δp]

# full 3-D world CoM positions of the trajectory knots (x̄[1:3] = 0)
P = [p.x̄[1:3] .+ X[k, 1:3] for k in 1:N]

# only render the lower portion of the descent (the final approach near/below the
# trigger altitude) so the tower and catch geometry can be shown up close.
const ZCUT = 0.650                       # highest altitude shown (km)
K0 = findfirst(k -> P[k][3] <= ZCUT, 1:N)

# ---------------------------------------------------------------------------
# pinhole camera: look-at frame + perspective divide.
# ---------------------------------------------------------------------------
struct Cam; eye::Vector{Float64}; r::Vector{Float64}; u::Vector{Float64}; f::Vector{Float64}; focal::Float64; end
function make_cam(eye, target, up, focal)
    f = normalize(target - eye)
    r = normalize(cross(f, up))
    u = cross(r, f)
    Cam(eye, r, u, f, focal)
end
# returns (screen_x, screen_y, depth). depth > 0 is in front of the camera.
function proj(cam::Cam, Pw)
    d  = Pw .- cam.eye
    cz = dot(d, cam.f)
    (cam.focal * dot(d, cam.r) / cz, cam.focal * dot(d, cam.u) / cz, cz)
end

# camera presets to compare. each is (eye, target, focal); up is +z. the
# preset is chosen with the RC_CAM environment variable and named into the
# output filename. front-left views keep the descent broadside (spread out);
# right-side / frontal views are included for comparison.
const CAM_PRESETS = Dict(
    "high_left"    => ([-0.360, 0.560, 0.560], [ 0.010, 0.030, 0.250], 10.5),  # high, front-left, looking down
    "high_left_r1" => ([-0.210, 0.638, 0.560], [ 0.010, 0.030, 0.250], 10.5),  # high_left rotated ~15° more frontal
    "high_left_r2" => ([-0.634, 0.505, 0.638], [ 0.010, 0.030, 0.250], 13.1),  # high_left rotated ~15° more side-on, zoomed out
    "low_left_up"  => ([-0.430, 0.660, 0.115], [ 0.010, 0.035, 0.300], 12.6),  # low ≈ mid-tower, looking up
    "vlow_left_up" => ([-0.380, 0.590, 0.030], [ 0.020, 0.050, 0.250], 12.0),  # very low (near ground), looking up
    "high_right"   => ([ 0.380, 0.560, 0.560], [-0.010, 0.030, 0.250], 10.5),  # high, front-right, looking down
    "low_right_up" => ([ 0.430, 0.660, 0.115], [-0.010, 0.035, 0.300], 12.6),  # low, front-right, looking up
    "frontal_down" => ([ 0.020, 0.760, 0.470], [ 0.005, 0.030, 0.240], 10.5),  # nearly head-on, looking down
)
const VIEW = get(ENV, "RC_CAM", "high_left_r2")
let (e, t, foc) = CAM_PRESETS[VIEW]
    global cam = make_cam(e, t, [0.0, 0.0, 1.0], foc)
end

const FLOORZ = 0.0

sx(Pw) = proj(cam, Pw)[1]
sy(Pw) = proj(cam, Pw)[2]
sd(Pw) = proj(cam, Pw)[3]
ff(v)  = string(round(v, digits = 6))

# ===========================================================================
# preamble: colours + styles (shared palette with the quad-gates plot)
# ===========================================================================
empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usetikzlibrary{arrows.meta}
\definecolor{ctraj0}{HTML}{2062A8}   % near trajectory (deep blue)
\definecolor{ctraj1}{HTML}{AAC4DE}   % far trajectory (pale blue)
\definecolor{cends}{HTML}{2B2F36}    % start / catch text
\definecolor{cgrid}{HTML}{C9CDD3}    % ground grid
\definecolor{ctower}{HTML}{8E949C}   % tower body fill (steel gray)
\definecolor{ctoweredge}{HTML}{383D44}
\definecolor{carm}{HTML}{565C64}     % catch arms fill (far arm)
\definecolor{carmedge}{HTML}{2B2F36} % catch arms edge
\definecolor{crfill}{HTML}{AEB4BB}   % rocket cylinder side fill
\definecolor{crtop}{HTML}{D6DADF}    % rocket cylinder lit top cap
\definecolor{credge}{HTML}{4A5058}   % rocket cylinder edge / engine base
\definecolor{cengine}{HTML}{D1495B}  % engine-gimbal line (red)
\definecolor{csimpl}{HTML}{AEB4BB}   % simplified orientation ticks (light gray)
\definecolor{cfunnel}{HTML}{2C6FB0}  % catch funnel walls
\definecolor{ctrig}{HTML}{E6A817}    % trigger-altitude top face
\pgfplotsset{
  compat=1.18,
  pview/.style={
    hide axis,
    scale only axis,
    axis equal image,
    clip=false,
  },
}
\tikzset{
  annot/.style={font=\footnotesize},
}
""")

# --- ground grid (perspective depth cue) -----------------------------------
grid_draws = String[]
# grid aligned to the tower (centred at x = 0, front face at y = 0, half-width
# 0.0078 ⊂ the central cell [-0.01, 0.01]): 2 cells on each side in x, one cell
# behind for the tower footprint and exactly 6 cells in front (+y).
gstep = 0.02
gxlo, gxhi, gylo, gyhi = -0.05, 0.05, -0.02, 0.12
for gx in gxlo:gstep:gxhi
    a, b = proj(cam, [gx, gylo, FLOORZ]), proj(cam, [gx, gyhi, FLOORZ])
    push!(grid_draws, "\\draw[cgrid, line width=0.3pt] (axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end
for gy in gylo:gstep:gyhi
    a, b = proj(cam, [gxlo, gy, FLOORZ]), proj(cam, [gxhi, gy, FLOORZ])
    push!(grid_draws, "\\draw[cgrid, line width=0.3pt] (axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end

# ===========================================================================
# catch funnel corridor: {y >= 0, |x| <= aw + slope*y}, extruded z ∈ [0, z_trig]
# top face shaded to mark the trigger altitude.
# ===========================================================================
const aw    = p.arm_half_width
const slope = tan(p.arm_opening_angle)
const YMAX  = 0.120                 # feasible-region wedge extends to the grid's front edge
const ZTRIG = p.trigger_altitude

hwid(y) = aw + slope * y
# footprint corners, ordered around the trapezoid (near tower → far)
foot = [
    [ hwid(0.0),  0.0],
    [-hwid(0.0),  0.0],
    [-hwid(YMAX), YMAX],
    [ hwid(YMAX), YMAX],
]
funnel_bot = [[c[1], c[2], FLOORZ] for c in foot]
funnel_top = [[c[1], c[2], ZTRIG] for c in foot]

scr(Pw) = (q = proj(cam, Pw); "(axis cs:$(ff(q[1])),$(ff(q[2])))")
quad(a, b, c, d) = "$(scr(a)) -- $(scr(b)) -- $(scr(c)) -- $(scr(d)) -- cycle"

# trigger-altitude top cap (orange): drawn UNDER the rockets. Filled + a bold
# horizontal outline that reads as a clean band even when viewed nearly edge-on.
trig_cap_draws = String[]
let poly = join([scr(c) for c in funnel_top], " -- ") * " -- cycle"
    push!(trig_cap_draws, "\\fill[ctrig, opacity=0.30] $poly;")
    push!(trig_cap_draws, "\\draw[ctrig, line width=1.1pt, line join=round] $poly;")
end

# blue feasible-region corridor: drawn ON TOP of everything else.
funnel_draws = String[]
# translucent side walls: the two slanted funnel walls (|x| = aw + slope·y) plus
# the far wall, so the trapezoidal keep-in volume reads. The near (tower-face)
# wall is left open so the catch stays clear.
for (i, j) in ((1, 4), (2, 3), (4, 3))   # +x wall, −x wall, far wall
    push!(funnel_draws, "\\fill[cfunnel, opacity=0.07] $(quad(funnel_bot[i], funnel_bot[j], funnel_top[j], funnel_top[i]));")
end
# vertical edges of the corridor (dashed)
for i in 1:4
    a, b = proj(cam, funnel_bot[i]), proj(cam, funnel_top[i])
    push!(funnel_draws, "\\draw[cfunnel, line width=1.0pt, dash pattern=on 3pt off 2pt] " *
        "(axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end
# bottom trapezoid outline (dashed)
let poly = join([scr(c) for c in funnel_bot], " -- ") * " -- cycle"
    push!(funnel_draws, "\\draw[cfunnel, line width=0.9pt, dash pattern=on 3pt off 2pt, line join=round] $poly;")
end

# ===========================================================================
# Mechazilla-style tower: a tall rectangular-prism body with two flat,
# trapezoidal chopstick arms angling out of the front face at the catch height.
# ===========================================================================
const TOWER_H  = p.tower_height
const CATCH_Z  = p.catch_altitude

# a solid rectangular prism, drawn as its camera-facing side faces + top cap,
# painter-sorted back-to-front, with edge outlines.
function box_prism(xlo, xhi, ylo, yhi, zlo, zhi; fillc, edgec, op = 1.0, ew = 0.6)
    cxy = [(xhi, yhi), (xhi, ylo), (xlo, ylo), (xlo, yhi)]
    bot = [[c[1], c[2], zlo] for c in cxy]
    top = [[c[1], c[2], zhi] for c in cxy]
    out = String[]
    faces  = [(i, mod1(i + 1, 4)) for i in 1:4]
    fdepth = [(sd(bot[a]) + sd(bot[b])) / 2 for (a, b) in faces]
    for k in sortperm(fdepth; rev = true)
        (a, b) = faces[k]
        q = quad(bot[a], bot[b], top[b], top[a])
        push!(out, "\\fill[$fillc, opacity=$op] $q;")
        push!(out, "\\draw[$edgec, line width=$(ew)pt, line join=round] $q;")
    end
    cap = join([scr(c) for c in top], " -- ") * " -- cycle;"
    push!(out, "\\fill[$fillc, opacity=$op] $cap")
    push!(out, "\\draw[$edgec, line width=$(ew)pt, line join=round] $cap")
    out
end

# a flat (thin) extruded plate given its base polygon corners + a z-thickness:
# used for the trapezoidal catch arms so they read as slim rectangular prisms.
function flat_prism(corners; τ, fillc, edgec, op = 1.0, ew = 0.5)
    bot = corners
    top = [c .+ [0.0, 0.0, τ] for c in corners]
    n   = length(corners)
    out = String[]
    push!(out, "\\fill[$fillc, opacity=$op] " * join([scr(c) for c in bot], " -- ") * " -- cycle;")
    for i in 1:n
        j = mod1(i + 1, n)
        q = quad(bot[i], bot[j], top[j], top[i])
        push!(out, "\\fill[$fillc, opacity=$op] $q;")
        push!(out, "\\draw[$edgec, line width=$(ew)pt, line join=round] $q;")
    end
    push!(out, "\\fill[$fillc, opacity=$op] " * join([scr(c) for c in top], " -- ") * " -- cycle;")
    push!(out, "\\draw[$edgec, line width=$(ew)pt, line join=round] " * join([scr(c) for c in top], " -- ") * " -- cycle;")
    out
end

# tower body: square-ish prism, front face at y = 0 (the corridor mouth), sitting
# just behind the catch point.
const TXW   = 0.0078                        # tower half-width in x
const TDEP  = 0.0150                         # tower depth in y (front face at y = 0)
tower_draws = box_prism(-TXW, TXW, -TDEP, 0.0, FLOORZ, TOWER_H;
                        fillc = "ctower", edgec = "ctoweredge", op = 1.0, ew = 0.5)

# two chopstick arms: flat trapezoidal plates that splay out of the front face on
# either side (±x) and forward (+y) at the catch altitude, angled slightly up.
const ARM_LEN  = 0.040      # forward reach (+y)
const ARM_ANG  = 0.62       # plan-view splay half-angle (rad) of each arm's axis
const ARM_WR   = 0.0075     # width at the root (near the tower)
const ARM_WT   = 0.0055     # width at the tip
const ARM_RISE = 0.002      # tip rises this much above the root (nearly level)
const ARM_THK  = 0.0065     # plate thickness / height (z)
const ARM_YBK = -TDEP       # arm root runs all the way back to the tower's back face
# an arm segment spanning y ∈ [ya, yb]; the arm axis, width taper (ARM_WR→ARM_WT)
# and slight rise are shared, so segments join seamlessly at the split.
function arm_prism(s, ya, yb; fillc = "carm", edgec = "carmedge")
    tanA   = s * tan(ARM_ANG)
    xin(y) = s * aw + tanA * y
    wof(y) = ARM_WR + (ARM_WT - ARM_WR) * (y - ARM_YBK) / (ARM_LEN - ARM_YBK)
    zof(y) = CATCH_Z + ARM_RISE * (y - ARM_YBK) / (ARM_LEN - ARM_YBK)
    corners = [[xin(ya), ya, zof(ya)], [xin(ya) + s * wof(ya), ya, zof(ya)],
               [xin(yb) + s * wof(yb), yb, zof(yb)], [xin(yb), yb, zof(yb)]]
    flat_prism(corners; τ = ARM_THK, fillc = fillc, edgec = edgec, op = 1.0, ew = 0.45)
end
# split each arm at the tower's front face (y = 0): the rear half is hidden behind
# the tower, the front half reaches toward the viewer in front of it. The nearer
# (−x) arm uses the tower's own gray and is painted last (on top of the blue
# keep-in lines); the far (+x) arm is drawn with the rest of the scene.
arm_back_draws      = vcat(arm_prism(-1, ARM_YBK, 0.0; fillc = "ctower", edgec = "ctoweredge"),
                           arm_prism(+1, ARM_YBK, 0.0))
arm_farfront_draws  = arm_prism(+1, 0.0, ARM_LEN)
arm_nearfront_draws = arm_prism(-1, 0.0, ARM_LEN; fillc = "ctower", edgec = "ctoweredge")

# ===========================================================================
# trajectory: per-segment polyline of the CoM, depth-shaded near-black (near) →
# light-gray (far), matching the quad-gates plot.
# ===========================================================================
traj_sx = [sx(P[k]) for k in 1:N]
traj_sy = [sy(P[k]) for k in 1:N]
zsh = [P[k][3] for k in 1:N]
zlo, zhi = extrema(zsh[K0:N])
# depth cue by height in the scene: near the catch (low, foreground) → dark,
# high up the descent (reads as distant) → light.
shade(z) = clamp((z - zlo) / (zhi - zlo), 0, 1)
lerpc(a, b, t) = round(Int, a + (b - a) * t)
graycol(c0, c1, z) = (t = shade(z);
    "{rgb,255:red,$(lerpc(c0[1],c1[1],t));green,$(lerpc(c0[2],c1[2],t));blue,$(lerpc(c0[3],c1[3],t))}")

traj_c0 = (43, 47, 54)          # dark  (low / near the catch)  #2B2F36
traj_c1 = (172, 178, 186)       # light (high / distant)        #ACB2BA
traj_draws = String[]
for k in K0:N-1
    col = graycol(traj_c0, traj_c1, (zsh[k] + zsh[k+1]) / 2)
    push!(traj_draws,
        "\\draw[line width=0.7pt, line cap=round, color=$col] " *
        "(axis cs:$(ff(traj_sx[k])),$(ff(traj_sy[k]))) -- (axis cs:$(ff(traj_sx[k+1])),$(ff(traj_sy[k+1])));")
end

# ===========================================================================
# rockets: a thin gray cylinder at a few select knots along the descent, each at
# its solved attitude, with a thin red engine-gimbal line out of the base.
# ===========================================================================
const RKT_H   = p.booster_height        # body length (true scale, ~71 m)
const RKT_R   = 1.00 * p.booster_radius # body radius (thin cylinder)
const ENG_LEN = 0.016                   # engine-gimbal line length (short stub)
function rocket_glyph(center, Q, δr, δp)
    zb  = Q * [0.0, 0.0, 1.0]           # nose direction (world)
    top = center .+ (RKT_H / 2) .* zb
    bot = center .- (RKT_H / 2) .* zb   # engine base
    r   = cam.focal * RKT_R / sd(center)
    pt  = collect(proj(cam, top)[1:2])
    pb  = collect(proj(cam, bot)[1:2])
    axs = normalize(pt .- pb); n = [-axs[2], axs[1]]
    # a body-frame circle of radius RKT_R about a world centre, projected to the
    # screen (an ellipse) — used for the two cylinder end caps so the body reads
    # as a solid 3-D tube rather than a flat bar.
    capp(cw) = join(["(axis cs:$(ff(q[1])),$(ff(q[2])))" for q in
        (proj(cam, cw .+ RKT_R .* (Q * [cos(t), sin(t), 0.0])) for t in range(0, 2π, length = 22))],
        " -- ") * " -- cycle"
    out = String[]
    # short red engine-gimbal stub, along the gimballed thrust axis, drawn FIRST
    # so it lies behind the rocket body (only the part below the base shows).
    etip = bot .+ ENG_LEN .* (-normalize(Q * rocket_thrust_axis_body(δr, δp)))
    a, b = proj(cam, bot), proj(cam, etip)
    push!(out, "\\draw[cengine, line width=1.3pt, line cap=round] " *
        "(axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
    # cylinder side (rectangle between the two caps' width tangents)
    body = join(["(axis cs:$(ff(q[1])),$(ff(q[2])))" for q in
                 (pb .+ r .* n, pt .+ r .* n, pt .- r .* n, pb .- r .* n)], " -- ") * " -- cycle"
    push!(out, "\\fill[crfill] $body;")
    push!(out, "\\fill[credge] $(capp(bot));")                       # dark engine base
    push!(out, "\\draw[credge, line width=0.5pt, line join=round] $body;")
    push!(out, "\\fill[crtop] $(capp(top));")                        # lit top cap
    push!(out, "\\draw[credge, line width=0.5pt, line join=round] $(capp(top));")
    out
end

# full rocket glyphs only at points of interest: the two knots where the booster
# orientation is most extreme (largest tilt from vertical), plus the catch (N).
bodytilt(k) = acos(clamp((qtoQ(rptoq(X[k, 4:6])) * [0.0, 0.0, 1.0])[3], -1.0, 1.0))
# extreme_ks = Int[]
# for (k, _) in sort([(k, bodytilt(k)) for k in K0:N-1]; by = t -> t[2], rev = true)
#     if all(abs(k - kk) >= 6 for kk in extreme_ks)
#         push!(extreme_ks, k)
#         length(extreme_ks) == 2 && break
#     end
# end
# rocket_ks = sort([extreme_ks; N])
rocket_ks=[N]
rocket_draws = String[]
for k in sort(rocket_ks; by = kk -> sd(P[kk]), rev = true)
    Q = qtoQ(rptoq(X[k, 4:6]))
    append!(rocket_draws, rocket_glyph(P[k], Q, X[k, 13], X[k, 14]))
end

# at interleaved intermediate knots (every 2nd knot, skipping the glyph knots), a
# simplified marker: a short tilted gray capsule-line along the booster axis that
# conveys location + orientation without the full glyph.
const L_SIMPL   = 0.55 * RKT_H       # shorter than the full glyph body
const ENG_SIMPL = 0.010              # short red gimbal stub for the simple marker
# depth-shaded gray by scene height: near the catch (low) darker, high up lighter.
rkt_c0 = (0x66, 0x6C, 0x74)          # low / near the catch (darker gray)
rkt_c1 = (0xCC, 0xD1, 0xD6)          # high / distant (light gray)
# marker knots spaced by a roughly constant arc length (a bit tighter than the
# old every-3rd-knot cadence up top): keeps the upper-descent density but thins
# the bunched, decelerating approach near the tower so markers don't overlap. The
# run is also stopped short of the catch so no marker touches the detailed glyph.
seglen = [norm(P[k+1] .- P[k]) for k in 1:N-1]
step_arc = 0.78 * sum(seglen[K0:min(K0 + 2, N - 1)])
const MARK_CUTOFF = 0.95 * RKT_H     # keep markers at least this far from the catch
marker_ks = Int[]
let acc = 0.0
    for k in K0:N-1
        norm(P[k] .- P[N]) < MARK_CUTOFF && break   # stop before the detailed catch glyph
        if isempty(marker_ks)
            push!(marker_ks, k)
        else
            acc += seglen[k-1]
            if acc >= step_arc
                push!(marker_ks, k); acc = 0.0
            end
        end
    end
end
glyph_set = Set(rocket_ks)
simpl_draws = String[]
for k in marker_ks
    k in glyph_set && continue
    Q   = qtoQ(rptoq(X[k, 4:6]))
    zb  = Q * [0.0, 0.0, 1.0]
    bot = P[k] .- (L_SIMPL / 2) .* zb   # engine base
    top = P[k] .+ (L_SIMPL / 2) .* zb
    # short thin red gimbal stub out of the base, drawn first (behind the body)
    et = bot .+ ENG_SIMPL .* (-normalize(Q * rocket_thrust_axis_body(X[k, 13], X[k, 14])))
    ea, eb = proj(cam, bot), proj(cam, et)
    push!(simpl_draws, "\\draw[cengine, line width=1.0pt, line cap=round] " *
        "(axis cs:$(ff(ea[1])),$(ff(ea[2]))) -- (axis cs:$(ff(eb[1])),$(ff(eb[2])));")
    # simplified body (opaque, so the trajectory line is hidden behind it), gray
    # shaded by distance to the camera.
    a, b = proj(cam, bot), proj(cam, top)
    push!(simpl_draws, "\\draw[color=$(graycol(rkt_c0, rkt_c1, P[k][3])), line width=1.8pt, line cap=round] " *
        "(axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end

# ===========================================================================
# labels
# ===========================================================================
const LBLBG = "fill=white, fill opacity=0.82, text opacity=1, inner sep=1.5pt, rounded corners=1pt"
labels = String[]
# trigger-altitude tag on the top face
let ctag = proj(cam, [hwid(YMAX), YMAX, ZTRIG])
    push!(labels, "\\node[annot, text=ctrig, $LBLBG, anchor=west, xshift=-11pt, yshift=14pt] " *
        "at (axis cs:$(ff(ctag[1])),$(ff(ctag[2]))) {\$h_{\\mathrm{trigger}}\$};")
end
# keep-in region constraint tag, low in the blue corridor (clear of the catch glyph)
let rtag = proj(cam, [0.0, 0.5 * YMAX, 0.010])
    push!(labels, "\\node[annot, text=cfunnel, $LBLBG, anchor=center, xshift=13pt, yshift=-10pt] " *
        "at (axis cs:$(ff(rtag[1])),$(ff(rtag[2]))) {\$A\\mathbf{x} \\le b\$};")
end

# phantom point to pad the top of the figure -------------------------------
all_sy = Float64[]
append!(all_sy, traj_sy[K0:N])
append!(all_sy, [sy(c) for c in funnel_top])
push!(all_sy, sy([0.0, -TDEP / 2, TOWER_H]))
top_pad = maximum(all_sy) + 0.06 * (maximum(all_sy) - minimum(all_sy))
bot_pad = minimum(all_sy) - 0.04 * (maximum(all_sy) - minimum(all_sy))

# ===========================================================================
# assemble (painter order: grid → arms → tower (on top of the arms) → CoM
#           trajectory → trigger cap → rockets → blue feasible region on top →
#           labels)
# ===========================================================================
ax = @pgf Axis(
    {
        "pview", width = "11cm", height = "9cm",
    },
    @pgf(Plot({draw = "none", mark = "none"}, Coordinates([sx(P[N])], [top_pad]))),
    @pgf(Plot({draw = "none", mark = "none"}, Coordinates([sx(P[N])], [bot_pad]))),
    grid_draws...,
    arm_back_draws...,      # rear halves of both arms (behind the tower)
    tower_draws...,
    arm_farfront_draws...,  # far (+x) arm, front half
    traj_draws...,
    simpl_draws...,         # simplified orientation ticks at intermediate knots
    trig_cap_draws...,      # orange z_trig cap: under the rockets
    rocket_draws...,
    funnel_draws...,        # blue feasible region
    arm_nearfront_draws..., # near (−x) arm on top of the blue keep-in lines
    labels...,
)

fig = @pgf TikzPicture(ax)

# default run writes the canonical name; an explicit RC_CAM (comparison) suffixes.
outbase = haskey(ENV, "RC_CAM") ? "rocket_catch_3d_persp_" * VIEW : "rocket_catch_3d_persp"
pgfsave(joinpath(@__DIR__, outbase * ".svg"), fig)
pgfsave(joinpath(@__DIR__, outbase * ".pdf"), fig)
@info "wrote $outbase (.pdf/.svg)  [view=$VIEW]"
