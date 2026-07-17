using PGFPlotsX, MarbleBenchmarks, Serialization, LinearAlgebra

# attitude helpers (reference quaternion is identity, so q = rptoq(δrp) and the
# body→world rotation is qtoQ(q)).
const rptoq = MarbleBenchmarks.rptoq
const qtoQ  = MarbleBenchmarks.qtoQ

# ===========================================================================
# Quadrotor-gates trajectory — slightly top-down, looking-forward PERSPECTIVE
# view (PGFPlotsX / TikZ). Same content as plot_tikz_quadrotor_gates_3d.jl but
# rendered through a pinhole camera so the gates read as receding 3-D frames.
#
# The projection is done here in Julia: every world point (trajectory knots,
# gate-frame corners, ground grid) is passed through a look-at + perspective
# camera to a 2-D screen coordinate, then emitted into a hidden-axis TikZ plot.
# ===========================================================================

# --- load the Marble solution ----------------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

prob = get_problem(:quadrotor_gates_3d)
inds, p = prob.inds, prob.params
N, ngates = p.N, p.ngates

z = deserialize(solfile)["quadrotor_gates_3d"]
X = z[inds[:x]]        # N × 9  (Δ-state; x̄[1:3] = 0 so cols 1:3 are world x,y,z)
μ = z[inds[:μ]]        # N × ngates

# full 3-D world positions of the trajectory knots
P = [p.x̄[1:3] .+ X[k, 1:3] for k in 1:N]

const MU_TOL = 1e-3
progress = [(k, argmax(μ[k, :])) for k in 1:N if maximum(μ[k, :]) > MU_TOL]

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

# slightly elevated, behind the start (y<0), looking forward along +y and down.
# pulled well back with a longer focal length so the depth compression is gentle
# (no single near gate blows up in size).
cam = make_cam([-5.5, -12.0, 8.0],  # eye (shifted left, so we face rightward)
               [-0.5,   5.0, 1.2],  # target (roughly the arena centre)
               [0.0, 0.0, 1.0],     # up
               5.0)                 # focal length (larger = narrower FOV)

# the drawn "floor" sits a bit below z = 0 so the gate ground-stems stay clear
# of the gate frames (one gate dips below z = 0).
const FLOORZ = -1.0

sx(Pw) = proj(cam, Pw)[1]
sy(Pw) = proj(cam, Pw)[2]
sd(Pw) = proj(cam, Pw)[3]
ff(v)  = string(round(v, digits = 5))

# --- gate frames as 3-D rectangles -----------------------------------------
# each gate spans its width axis (local x = Rz[:,1]) and its height axis (z),
# ±half-dims about the centre. dimensions are inflated by a small factor purely
# for legibility (frame + posts both follow, since they use these corners).
const GATE_INFLATE = 1.6
hw, hh = GATE_INFLATE * p.gate_dims[1] / 2, GATE_INFLATE * p.gate_dims[3] / 2
gate_corners = map(1:ngates) do j
    c   = p.gate_positions[j]
    wax = p.gate_orientations[j][:, 1]     # width direction (in xy-plane)
    zax = [0.0, 0.0, 1.0]                   # height direction
    [c .+ hw*wax .+ hh*zax,
     c .- hw*wax .+ hh*zax,
     c .- hw*wax .- hh*zax,
     c .+ hw*wax .- hh*zax]
end
gate_depth = [sd(p.gate_positions[j]) for j in 1:ngates]   # for painter's sort

# ===========================================================================
# preamble: colours + styles (shared with the top-down plot)
# ===========================================================================
const GATE_HEX = ["2C6FB0", "E6A817", "2E9E52", "CB4154"]

empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usetikzlibrary{arrows.meta}
\definecolor{ctraj}{HTML}{2B2F36}    % trajectory
\definecolor{cknot}{HTML}{6E747C}    % highlighted γ knots
\definecolor{cends}{HTML}{2B2F36}    % start / goal text
\definecolor{cgrid}{HTML}{C9CDD3}    % ground grid
\definecolor{cqbody}{HTML}{3A3F47}   % quad arms / hub
\definecolor{cqfill}{HTML}{C3C8CE}   % quad rotor discs
\definecolor{cqedge}{HTML}{6E747C}   % quad rotor rims
""" *
    join(["\\definecolor{g$j}{HTML}{$(GATE_HEX[j])}" for j in 1:ngates], "\n") *
raw"""
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
gxlo, gxhi, gylo, gyhi = -5.0, 3.0, -1.0, 11.0
for gx in gxlo:1.0:gxhi
    a, b = proj(cam, [gx, gylo, FLOORZ]), proj(cam, [gx, gyhi, FLOORZ])
    push!(grid_draws, "\\draw[cgrid, line width=0.3pt] (axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end
for gy in gylo:1.0:gyhi
    a, b = proj(cam, [gxlo, gy, FLOORZ]), proj(cam, [gxhi, gy, FLOORZ])
    push!(grid_draws, "\\draw[cgrid, line width=0.3pt] (axis cs:$(ff(a[1])),$(ff(a[2]))) -- (axis cs:$(ff(b[1])),$(ff(b[2])));")
end

# --- gate frames, painted far-to-near --------------------------------------
gate_draws = String[]
for j in sortperm(gate_depth; rev = true)
    cs = gate_corners[j]
    pts = [proj(cam, c) for c in cs]
    poly = join(["(axis cs:$(ff(q[1])),$(ff(q[2])))" for q in pts], " -- ") * " -- cycle"
    push!(gate_draws, "\\fill[g$j, opacity=0.14] $poly;")
    push!(gate_draws, "\\draw[draw=g$j, line width=2.2pt, line join=round] $poly;")
    # two solid legs from the floor up to the gate's bottom corners, anchoring
    # the frame in 3-D (same colour as the gate border).
    for corner in (cs[3], cs[4])          # bottom-left, bottom-right
        top = proj(cam, corner)
        flo = proj(cam, [corner[1], corner[2], FLOORZ])
        push!(gate_draws, "\\draw[g$(j), line width=1.0pt] (axis cs:$(ff(flo[1])),$(ff(flo[2]))) -- (axis cs:$(ff(top[1])),$(ff(top[2])));")
    end
end

# --- trajectory: per-segment polyline, coloured by distance to the camera —
#     nearest is near-black, farthest is light gray (a depth cue). ----------
traj_sx = [sx(P[k]) for k in 1:N]
traj_sy = [sy(P[k]) for k in 1:N]
traj_d  = [sd(P[k]) for k in 1:N]      # depth along the view axis
dmin, dmax = extrema(traj_d)

traj_c0 = (172, 178, 186)   # nearest  #ACB2BA (light gray)
traj_c1 = (43, 47, 54)      # farthest #2B2F36 (near-black)
lerpc(a, b, t) = round(Int, a + (b - a) * t)
traj_draws = String[]
for k in 1:N-1
    dmid = (traj_d[k] + traj_d[k+1]) / 2
    t = (dmid - dmin) / (dmax - dmin)
    R = lerpc(traj_c0[1], traj_c1[1], t)
    G = lerpc(traj_c0[2], traj_c1[2], t)
    B = lerpc(traj_c0[3], traj_c1[3], t)
    push!(traj_draws,
        "\\draw[line width=1.2pt, line cap=round, color={rgb,255:red,$R;green,$G;blue,$B}] " *
        "(axis cs:$(ff(traj_sx[k])),$(ff(traj_sy[k]))) -- (axis cs:$(ff(traj_sx[k+1])),$(ff(traj_sy[k+1])));")
end

# --- progress knots: a little quadrotor glyph, projected through the camera —
#     centre hub + four X-arms + four flat rotor discs (neutral gray so it
#     stands out against every gate). ---------------------------------------
# Q is the body→world rotation at this knot; the quad's arm/rotor plane is the
# body x–y plane, so all body geometry is rotated by Q before projecting.
function quad_glyph(center, Q; Larm = 0.26, rrot = 0.12, rhub = 0.05, ndisc = 24)
    bdirs = [normalize([1.0, 1.0, 0.0]), normalize([-1.0, 1.0, 0.0]),
             normalize([-1.0, -1.0, 0.0]), normalize([1.0, -1.0, 0.0])]
    # rotor disc lying in the body plane, projected (tilts with attitude)
    hdisc(cnt, r) = join(["(axis cs:$(ff(q[1])),$(ff(q[2])))" for q in
                          (proj(cam, cnt .+ r .* (Q * [cos(t), sin(t), 0.0])) for t in range(0, 2π, length = ndisc))],
                         " -- ") * " -- cycle"
    # camera-facing disc (a sphere always projects to a round blob)
    bdisc(scr, r) = join(["(axis cs:$(ff(scr[1] + r*cos(t))),$(ff(scr[2] + r*sin(t))))"
                          for t in range(0, 2π, length = ndisc)], " -- ") * " -- cycle"
    out  = String[]
    cscr = proj(cam, center)
    rcs  = [center .+ Larm .* (Q * d) for d in bdirs]
    for rc in rcs                       # arms first, rotors land on the tips
        a = proj(cam, rc)
        push!(out, "\\draw[cqbody, line width=1.1pt, line cap=round] " *
                   "(axis cs:$(ff(cscr[1])),$(ff(cscr[2]))) -- (axis cs:$(ff(a[1])),$(ff(a[2])));")
    end
    push!(out, "\\fill[cqbody] $(bdisc(cscr, cam.focal * rhub / sd(center)));")  # sphere hub
    for rc in rcs                       # flat rotor discs (drawn above the hub)
        poly = hdisc(rc, rrot)
        push!(out, "\\fill[cqfill, fill opacity=0.92] $poly;")
        push!(out, "\\draw[cqedge, line width=0.8pt] $poly;")
    end
    out
end

# orientation for the glyph: keep the bank (body-up) from the solved attitude,
# but yaw the heading to face the knot's velocity.
function facing_frame(Q, v)
    zb = normalize(Q * [0.0, 0.0, 1.0])          # body-up in world (the bank)
    vh = [v[1], v[2], 0.0]                        # horizontal velocity
    norm(vh) < 1e-6 && (vh = copy(v))
    fwd = vh - dot(vh, zb) * zb                   # velocity projected into rotor plane
    norm(fwd) < 1e-9 && return Q                  # degenerate: fall back to attitude
    fwd = normalize(fwd)
    hcat(fwd, cross(zb, fwd), zb)                 # [forward, left, up], body→world
end

quad_draws = String[]
for (k, _) in progress
    Q = qtoQ(rptoq(X[k, 4:6]))
    R = facing_frame(Q, X[k, 7:9])
    append!(quad_draws, quad_glyph(P[k], R))
end

prog_labels = [
    "\\node[annot, text=g$j, rotate=12, anchor=south west, inner sep=1.5pt, " *
    "fill=white, fill opacity=0.85, text opacity=1, rounded corners=1pt] " *
    "at (axis cs:$(ff(sx(P[k]))),$(ff(sy(P[k])))) {\$\\mu^{($j)} = 1\$};"
    for (k, j) in progress
]

# --- three highlighted γ knots ---------------------------------------------
kmid = 40
gamma_knots = [
    (1,    "\\gamma = (0,0,0,0)", "east"),
    (kmid, "\\gamma = (1,1,1,0)", "east"),
    (N,    "\\gamma = (1,1,1,1)", "west"),
]
gamma_ks = [k for (k, _, _) in gamma_knots]
gamma_labels = [
    "\\node[annot, cends, anchor=$anch, inner sep=2pt, fill=white, " *
    "fill opacity=0.85, text opacity=1, rounded corners=1pt] " *
    "at (axis cs:$(ff(sx(P[k]))),$(ff(sy(P[k])))) {\$$lbl\$};"
    for (k, lbl, anch) in gamma_knots
]

# --- small gate index at each gate's (screen) top-left corner ---------------
gate_num_labels = String[]
for j in 1:ngates
    pts = [proj(cam, c) for c in gate_corners[j]]
    tl  = pts[argmax([q[2] - q[1] for q in pts])]   # up-and-left corner in screen space
    push!(gate_num_labels,
        "\\node[text=g$j, font=\\small\\bfseries, anchor=north west, inner sep=1.5pt, xshift=3pt, yshift=-3pt] " *
        "at (axis cs:$(ff(tl[1])),$(ff(tl[2]))) {$j};")
end

# ===========================================================================
# assemble
# ===========================================================================
ax = @pgf Axis(
    {
        "pview", width = "11cm", height = "8cm",
    },
    grid_draws...,
    traj_draws...,
    @pgf(Plot({only_marks, mark = "*", mark_size = "2.4pt",
               mark_options = "{fill=cknot, draw=white, line width=0.5pt}"},
              Coordinates([sx(P[k]) for k in gamma_ks], [sy(P[k]) for k in gamma_ks]))),
    quad_draws...,
    gate_draws...,          # gates painted last, on top of the quadrotors
    gamma_labels...,
    prog_labels...,
    "\\node[annot, cends, anchor=north, yshift=-2pt] at (axis cs:$(ff(sx(P[1]))),$(ff(sy(P[1])))) {start};",
    gate_num_labels...,
)

fig = @pgf TikzPicture(ax)

outsvg = joinpath(@__DIR__, "quadrotor_gates_3d_persp.svg")
pgfsave(outsvg, fig)
pgfsave(joinpath(@__DIR__, "quadrotor_gates_3d_persp.pdf"), fig)
@info "wrote $outsvg  ($(length(progress)) progress knots: $(progress))"
