using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie.GeometryBasics: Cylinder
import GLMakie.Makie as MK      # rich (multi-colour) text + colour helpers

# ===========================================================================
# Quadrotor-gates trajectory — GLMakie animation (mp4).
#
# Animated companion to plot_glmakie_quadrotor_gates_3d.jl: the same scene
# (four coloured gates + shaded quadrotor mesh + ground grid), but instead of a
# strobe of quadrotors at fixed knots, a single quadrotor flies the Marble
# solution. The state is interpolated between knots (position/velocity linearly,
# attitude by slerp) so the motion is smooth at the video frame rate.
#
# Extra behaviour requested:
#   * the flight PAUSES for 1 s each time a gate is cleared, i.e. each knot where
#     that gate's progress multiplier μ[k, j] jumps above zero.
#   * the quadrotor is coloured by the most recent gate it has flown through
#     (neutral slate before the first gate). Because of the waypoint_order
#     constraint the gates are always cleared in order 1→2→3→4, so the most
#     recent gate index is just the number of gates cleared so far.
#   * a γ = (a, b, c, d) label follows the quadrotor, showing which gates have
#     been cleared. Each "1" is drawn in its own gate's colour; the "0"s and the
#     surrounding punctuation are a muted gray.
# ===========================================================================

GLMakie.activate!()
GLMakie.closeall() # close any open screen

# attitude helpers (reference quaternion is identity, so q = rptoq(δrp) is the
# body attitude quaternion directly and qtoQ(q) is the body→world rotation).
const rptoq = MarbleBenchmarks.rptoq
const qtoQ  = MarbleBenchmarks.qtoQ

# --- load the latest Marble solution ---------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

problem = get_problem(:quadrotor_gates_3d)
inds, p = problem.inds, problem.params
N, ngates = p.N, p.ngates

z = deserialize(solfile)["quadrotor_gates_3d"]
X = z[inds[:x]]        # N × 9  (Δ-state; x̄[1:3] = 0 so cols 1:3 are world x,y,z)
μ = z[inds[:μ]]        # N × ngates  (progress multipliers; spikes as a gate is cleared)

# Reconstruct the world-frame trajectory ------------------------------------
r_traj = [p.x̄[1:3] .+ X[k, 1:3] for k in 1:N]   # position
q_traj = [rptoq(X[k, 4:6]) for k in 1:N]         # body attitude quaternion (scalar-first)
v_traj = [X[k, 7:9] for k in 1:N]                # velocity

# knot at which each gate is cleared (μ spikes to ≈1 at a single knot), paired
# with its gate and sorted into flight order.
const MU_TOL = 1e-3
gate_pass = sort([(argmax(μ[:, j]), j) for j in 1:ngates])   # [(knot, gate), …]
pass_at   = Dict(k => j for (k, j) in gate_pass)             # knot → gate cleared there
PASS_S    = Float64[k for (k, _) in gate_pass]               # knots as float, for s-compares

# distinct per-gate palette (shared with plot_glmakie_quadrotor_gates_3d.jl)
const GATE_COLORS = ["#2C6FB0", "#E6A817", "#2E9E52", "#CB4154"]
const gate_cols   = [MK.to_color(c) for c in GATE_COLORS]

const NEUTRAL    = RGBf(0.68, 0.73, 0.80)   # quad colour before any gate is cleared
const ZERO_COLOR = RGBf(0.62, 0.65, 0.70)   # a "0" digit in the γ label
const PUNCT      = RGBf(0.20, 0.20, 0.20)   # "γ = (", commas and ")"

# --- quaternion helpers (scalar-first, matching L/qtoQ) --------------------
# scalar-first quaternion multiply: q = a ⊗ b
function qmul(a, b)
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    [w1*w2 - x1*x2 - y1*y2 - z1*z2,
     w1*x2 + x1*w2 + y1*z2 - z1*y2,
     w1*y2 - x1*z2 + y1*w2 + z1*x2,
     w1*z2 + x1*y2 - y1*x2 + z1*w2]
end

# shortest-path spherical interpolation between two scalar-first quaternions
function qslerp(qa, qb, t)
    d = dot(qa, qb)
    if d < 0            # take the shorter arc
        qb = -qb; d = -d
    end
    if d > 0.9995       # nearly parallel → normalized lerp avoids a 0/0
        q = qa .+ t .* (qb .- qa)
        return q ./ norm(q)
    end
    θ0 = acos(clamp(d, -1, 1))
    q2 = qb .- qa .* d
    q2 = q2 ./ norm(q2)
    return qa .* cos(θ0 * t) .+ q2 .* sin(θ0 * t)
end

# rotation matrix → scalar-first quaternion (Shepperd's method), for the gates
function Q_to_quat(R)
    tr = R[1,1] + R[2,2] + R[3,3]
    if tr > 0
        S = sqrt(tr + 1.0) * 2
        w = 0.25 * S; x = (R[3,2] - R[2,3]) / S; y = (R[1,3] - R[3,1]) / S; z = (R[2,1] - R[1,2]) / S
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        S = sqrt(1.0 + R[1,1] - R[2,2] - R[3,3]) * 2
        w = (R[3,2] - R[2,3]) / S; x = 0.25 * S; y = (R[1,2] + R[2,1]) / S; z = (R[1,3] + R[3,1]) / S
    elseif R[2,2] > R[3,3]
        S = sqrt(1.0 + R[2,2] - R[1,1] - R[3,3]) * 2
        w = (R[1,3] - R[3,1]) / S; x = (R[1,2] + R[2,1]) / S; y = 0.25 * S; z = (R[2,3] + R[3,2]) / S
    else
        S = sqrt(1.0 + R[3,3] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S; x = (R[1,3] + R[3,1]) / S; y = (R[2,3] + R[3,2]) / S; z = 0.25 * S
    end
    [w, x, y, z]
end

# scalar-first quaternion → GLMakie.Quaternion (scalar-last / (x,y,z,w))
makie_quat(q) = GLMakie.Quaternion(q[2], q[3], q[4], q[1])

# --- state interpolation at a continuous knot parameter s ∈ [1, N] ----------
function traj_at(s::Real)
    s = clamp(float(s), 1.0, float(N))
    k0 = floor(Int, s); k1 = min(k0 + 1, N); α = s - k0
    pos  = (1 - α) .* r_traj[k0] .+ α .* r_traj[k1]
    vel  = (1 - α) .* v_traj[k0] .+ α .* v_traj[k1]
    quat = qslerp(q_traj[k0], q_traj[k1], α)
    return pos, quat, vel
end

# number of gates cleared by parameter s (gates are cleared in order, so this is
# also the index of the most recent gate)
gates_cleared(s::Real) = count(pk -> pk <= s, PASS_S)

# --- meshes ----------------------------------------------------------------
quad = load(joinpath(@__DIR__, "assets", "quadrotor.obj"))
gate = load(joinpath(@__DIR__, "assets", "dronegate.obj"))

# shaded material with a restrained specular so the bright gate colours don't
# blow out to white. Scene brightness is handled by the lights (see below).
const LIT = (; shading=true, specular=0.1f0, shininess=12.0f0)
const QUAD_S = 0.075   # quadrotor mesh scale

# position/orient/colour the (single, reused) quadrotor mesh at a state
function set_quad!(quad_plot, pos, q, v)
    GLMakie.translate!(quad_plot, pos...)
    # yaw the heading toward the velocity, keeping the solved bank (attitude q).
    yaw  = norm(v) > 1e-6 ? atan(v[2], v[1]) : 0.0
    qyaw = [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]   # RotZ(yaw), scalar-first
    GLMakie.rotate!(quad_plot, makie_quat(qmul(qyaw, q)))
end

function plot_gate!(ax; pos, R=Matrix(I(3)), num=1, color=:slateblue, textc=:gray, s=0.85)
    gate_plot = mesh!(ax, gate; color=color, LIT...)
    GLMakie.scale!(gate_plot, (s, s, s))
    GLMakie.translate!(gate_plot, pos...)

    q = Q_to_quat(R)
    GLMakie.rotate!(gate_plot, makie_quat(q))

    # gate index label, laid into the gate plane (rotate the text upright first)
    q_text = qmul(q, [cos(π/4), sin(π/4), 0.0, 0.0])   # · RotX(π/2)
    text!(ax, string(num);
        position    = Point3f(pos...),
        space       = :data,
        align       = (:center, :center),
        fontsize    = s * p.gate_dims[1] / 6,
        color       = textc,
        alpha       = 0.9,
        markerspace = :data,
        rotation    = makie_quat(q_text),
    )

    # gate stilts down to the floor (skip if a corner sits below ground)
    for sgn in (-1.0, 1.0)
        corner = pos + R * (2.0 .* [s * sgn * 0.85, 0, -s] .* p.gate_dims / 2)
        corner[3] <= FLOORZ && continue
        bottom = Point3f((corner .* [1, 1, 0] .+ [0, 0, FLOORZ])...)
        top    = Point3f(corner...)
        mesh!(ax, Cylinder(bottom, top, 0.02f0); color=color, LIT...)
    end
    return nothing
end

# γ = (a, b, c, d) as multi-coloured rich text: each cleared gate's "1" in that
# gate's colour, the "0"s and punctuation muted.
function gamma_rich(γ)
    spans = Any["γ = ("]
    for j in 1:ngates
        col = γ[j] == 1 ? gate_cols[j] : ZERO_COLOR
        push!(spans, MK.rich(string(γ[j]); color=col, font=:bold))
        j < ngates && push!(spans, ", ")
    end
    push!(spans, ")")
    return MK.rich(spans...; color=PUNCT)
end

# simple ground grid at height z
function xy_grid!(ax; xlim, ylim, z, step=1.0, color=(:black, 0.1), lw=1)
    segs = Point3f[]
    for x in xlim[1]:step:xlim[2]
        push!(segs, Point3f(x, ylim[1], z)); push!(segs, Point3f(x, ylim[2], z))
    end
    for y in ylim[1]:step:ylim[2]
        push!(segs, Point3f(xlim[1], y, z)); push!(segs, Point3f(xlim[2], y, z))
    end
    linesegments!(ax, segs; color=color, linewidth=lw)
end

# --- scene bounds ----------------------------------------------------------
const xmin, xmax = -6.0, 3.0
const ymin, ymax = -1.0, 11.0
const zmin, zmax = -1.0, 3.0
const FLOORZ = -1.0

# --- animation timing -------------------------------------------------------
const FPS     = 60          # video frame rate (high, for smooth motion)
const SLOWMO  = 1.0         # motion runs this × real time (1.0 = the 9 s solution)
const PAUSE_S = 1.0         # freeze this long each time a gate is cleared
const START_S = 0.6         # short hold on the initial pose
const END_S   = 1.2         # hold on the final pose

frames_per_knot = max(1, round(Int, SLOWMO * p.Δt * FPS))
hold_frames     = round(Int, PAUSE_S * FPS)

# frame schedule: a list of continuous knot parameters s, with the gate-clearing
# knots repeated for the 1 s pause.
s_frames = Float64[]
append!(s_frames, fill(1.0, round(Int, START_S * FPS)))
for k in 2:N
    for f in 1:frames_per_knot
        push!(s_frames, (k - 1) + f / frames_per_knot)   # advance up to exactly k
    end
    haskey(pass_at, k) && append!(s_frames, fill(float(k), hold_frames))
end
append!(s_frames, fill(float(N), round(Int, END_S * FPS)))
const n_frames = length(s_frames)

# --- build the scene --------------------------------------------------------
lims = (xmin, xmax, ymin, ymax, zmin, zmax)

fig = Figure(size=(900, 700), figure_padding=0, backgroundcolor=:white)
ax  = Axis3(fig[1, 1], aspect=:data, perspectiveness=0.8, limits=lims)
hidedecorations!(ax); hidespines!(ax)

# brighten the scene: a strong ambient keeps the bright gate colours saturated,
# while the default camera-following directional light gives gentle 3-D shading.
set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

# gates — each in its own colour (frame + legs + number)
for i in 1:ngates
    plot_gate!(ax; pos=Point3f(Float32.(p.gate_positions[i])...),
               R=p.gate_orientations[i], num=i, color=GATE_COLORS[i], textc=GATE_COLORS[i])
end

# growing flown path behind the quadrotor
trail = Observable(Point3f[Point3f(r_traj[1]...)])
lines!(ax, trail; color=:black, joinstyle=:round, linecap=:round,
       linewidth=1.2, alpha=0.8)

# the single moving quadrotor (scale is fixed; translate/rotate/colour update)
qcolor    = Observable{Any}(NEUTRAL)
quad_plot = mesh!(ax, quad; color=qcolor, transparency=false, LIT...)
GLMakie.scale!(quad_plot, (QUAD_S, QUAD_S, QUAD_S))
let (pos, q, v) = traj_at(1.0)
    set_quad!(quad_plot, pos, q, v)
end

# γ label that follows the quadrotor (pixel offset so it clears the mesh)
gpos       = Observable(Point3f(r_traj[1]...))
gamma_text = Observable{Any}(gamma_rich(zeros(Int, ngates)))
text!(ax, gpos; text=gamma_text, space=:data, align=(:left, :bottom),
      offset=(16, 12), fontsize=22, rotation=deg2rad(15),
      strokecolor=RGBAf(1, 1, 1, 0.85), strokewidth=2.5,
      transparency=true, depth_shift=-1.0f0)

# ground grid
xy_grid!(ax; xlim=(xmin, xmax), ylim=(ymin, ymax), z=FLOORZ, step=1.0,
         color=(:black, 0.15), lw=1.5)

# low, looking-forward view matching the still.
ax.azimuth[]   = 4.428
ax.elevation[] = 0.366

# --- record -----------------------------------------------------------------
const OUTPATH = joinpath(@__DIR__, "animations", "quadrotor_gates_3d_glmakie.mp4")
mkpath(dirname(OUTPATH))

record(fig, OUTPATH, 1:n_frames; framerate=FPS) do i
    s = s_frames[i]
    pos, q, v = traj_at(s)
    set_quad!(quad_plot, pos, q, v)

    ncleared = gates_cleared(s)
    qcolor[] = ncleared == 0 ? NEUTRAL : gate_cols[ncleared]

    γ = [j <= ncleared ? 1 : 0 for j in 1:ngates]
    gamma_text[] = gamma_rich(γ)
    gpos[]       = Point3f(pos...)

    kf  = floor(Int, s)
    pts = Point3f[Point3f(r_traj[m]...) for m in 1:kf]
    push!(pts, Point3f(pos...))
    trail[] = pts
end

@info "saved animation" OUTPATH n_frames FPS
