using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie.GeometryBasics: Cylinder
using GLMakie.Makie.LaTeXStrings: latexstring   # annotations are typeset as LaTeX

# ===========================================================================
# Quadrotor-gates trajectory — *progress* variant of
# plot_glmakie_quadrotor_gates_3d.jl.
#
# Same scene and meshes, but the emphasis is on the complementarity progress
# variables instead of on the gates themselves:
#   * the gates (frame, legs and index label) are drawn in a flat, opaque
#     light blue-gray so they read as scenery and recede behind the flight path.
#   * quadrotors are highlighted in red at every knot where a gate's progress
#     multiplier is nonzero (μ[k, j] > 0) — i.e. the knots at which the vehicle is
#     actually passing through a gate. Each is annotated with the μ^(j) that fired.
#   * only three neutral gray quadrotors are drawn, one per labelled phase of the
#     flight: the start, the coast between the last two gates and the end. Each
#     carries the cumulative progress vector γ_k = (λ[k, 1], …), which steps from
#     (0,…,0) at the start to (1,…,1) once every gate is cleared.
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

# --- progress bookkeeping ---------------------------------------------------
# μ[k, j] is the progress the trajectory makes on gate j at knot k; it is zero
# everywhere except where the vehicle flies through the gate. λ (= γ in the
# figure) is its running sum, and is constrained to reach 1 for every gate.
const MU_TOL = 1e-3
λ = [sum(μ[1:k, j]) for k in 1:N, j in 1:ngates]

# knots with nonzero progress, paired with the gate(s) that progressed there
pass_knot = [(k, [j for j in 1:ngates if μ[k, j] > MU_TOL])
             for k in 1:N if maximum(μ[k, :]) > MU_TOL]
pass_ks   = Set(k for (k, _) in pass_knot)

# γ_k, the integer cumulative progress vector, is constant between gate passes;
# its plateaus are the phases of the flight.
γ_int(k) = clamp.(round.(Int, λ[k, :]), 0, 1)

function gamma_plateaus()
    runs, start = UnitRange{Int}[], 1
    for k in 2:N
        γ_int(k) == γ_int(start) && continue
        push!(runs, start:k-1); start = k
    end
    push!(runs, start:N)
    return runs
end

# The neutral (gray) quadrotors are only there to carry a γ label, so draw just
# three of them: the start (γ = 0), the end (γ = 1) and the coast between the last
# two gates (γ = (1,…,1,0)). The coast one is taken past the middle of its plateau,
# where the vehicle has pulled clear of the gate it just flew through.
const COAST_FRAC  = 0.7
const COAST_SHIFT = 0      # knots further along than that, to sit clear of gate 4

plateaus = gamma_plateaus()
coast    = findlast(r -> sum(γ_int(first(r))) == ngates - 1, plateaus)
along(r, f) = clamp(first(r) + round(Int, f * (last(r) - first(r))) + COAST_SHIFT,
                    first(r), last(r))
shown_ks = [k for k in unique([1
                               coast === nothing ? Int[] : along(plateaus[coast], COAST_FRAC)
                               N])
            if !(k in pass_ks)]

# --- palette ----------------------------------------------------------------
const GATE_ALPHA = 0.45                                    # gates are see-through scenery
const GATE_COLOR = RGBAf(0.780, 0.816, 0.859, GATE_ALPHA)  # flat light blue-gray
const GATE_TEXTC = RGBAf(0.612, 0.663, 0.729, GATE_ALPHA)  # index numbers, a shade darker
# the quadrotor colours are shared with the hopper contact plot, so the two figures
# read the same way: see-through slate for the knots that are just passing through,
# opaque light crimson for the knots where the complementarity is active, and the
# bright red reserved for the annotations sitting on top of them.
const QUAD_COLOR = RGBAf(0.68, 0.73, 0.80, 0.40)   # intermediate: transparent slate
const PASS_COLOR = RGBf(0.890, 0.604, 0.627)       # passing a gate: light crimson
const LABEL_RED  = RGBf(0.898, 0.106, 0.145)       # μ annotations
const LABEL_DARK = RGBf(0.15, 0.15, 0.15)      # γ annotations

# --- quaternion helpers (scalar-first, matching L/qtoQ) --------------------
# scalar-first quaternion multiply: q = a ⊗ b
function qmul(a, b)
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    [w1*w2 - x1*x2 - y1*y2 - z1*z2,
     w1*x2 + x1*w2 + y1*z2 - z1*y2,
     w1*y2 - x1*z2 + y1*w2 + z1*x2,
     w1*z2 + x1*y2 - y1*x2 + z1*w2]
end

# rotation matrix → scalar-first quaternion (Shepperd's method)
function Q_to_quat(R)
    tr = R[1,1] + R[2,2] + R[3,3]
    if tr > 0
        S = sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[3,2] - R[2,3]) / S
        y = (R[1,3] - R[3,1]) / S
        z = (R[2,1] - R[1,2]) / S
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        S = sqrt(1.0 + R[1,1] - R[2,2] - R[3,3]) * 2
        w = (R[3,2] - R[2,3]) / S
        x = 0.25 * S
        y = (R[1,2] + R[2,1]) / S
        z = (R[1,3] + R[3,1]) / S
    elseif R[2,2] > R[3,3]
        S = sqrt(1.0 + R[2,2] - R[1,1] - R[3,3]) * 2
        w = (R[1,3] - R[3,1]) / S
        x = (R[1,2] + R[2,1]) / S
        y = 0.25 * S
        z = (R[2,3] + R[3,2]) / S
    else
        S = sqrt(1.0 + R[3,3] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = (R[1,3] + R[3,1]) / S
        y = (R[2,3] + R[3,2]) / S
        z = 0.25 * S
    end
    [w, x, y, z]
end

# scalar-first quaternion → GLMakie.Quaternion (scalar-last / (x,y,z,w))
makie_quat(q) = GLMakie.Quaternion(q[2], q[3], q[4], q[1])

# --- meshes ----------------------------------------------------------------
quad = load(joinpath(@__DIR__, "assets", "quadrotor.obj"))
gate = load(joinpath(@__DIR__, "assets", "dronegate.obj"))

# shaded material with a restrained specular. Scene brightness is handled by the
# lights (see below).
const LIT = (; shading=true, specular=0.1f0, shininess=12.0f0)

# the gates are pure scenery here: kill the specular highlight so they stay a
# flat, even light gray instead of picking up a shaded 3-D sheen.
const FLAT = (; shading=true, specular=0.0f0, shininess=1.0f0)

function plot_quadrotor!(ax; p, q, v, color=QUAD_COLOR, s=0.075, transparency=false)
    quad_plot = mesh!(ax, quad; color=color, transparency=transparency, LIT...)

    GLMakie.scale!(quad_plot, (s, s, s))
    GLMakie.translate!(quad_plot, p...)

    # yaw the heading toward the velocity, keeping the solved bank (attitude q).
    yaw = norm(v) > 1e-6 ? atan(v[2], v[1]) : 0.0
    qyaw = [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]   # RotZ(yaw), scalar-first
    GLMakie.rotate!(quad_plot, makie_quat(qmul(qyaw, q)))

    return quad_plot
end

function plot_gate!(ax; pos, R=Matrix(I(3)), num=1, color=GATE_COLOR, textc=GATE_TEXTC, s=0.85)
    gate_plot = mesh!(ax, gate; color=color, transparency=true, FLAT...)
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
        markerspace = :data,
        rotation    = makie_quat(q_text),
    )

    # gate stilts down to the floor (skip if a corner sits below ground)
    for sgn in (-1.0, 1.0)
        corner = pos + R * (2.0 .* [s * sgn * 0.85, 0, -s] .* p.gate_dims / 2)
        corner[3] <= FLOORZ && continue
        bottom = Point3f((corner .* [1, 1, 0] .+ [0, 0, FLOORZ])...)
        top    = Point3f(corner...)
        mesh!(ax, Cylinder(bottom, top, 0.02f0); color=color, transparency=true, FLAT...)
    end
    return nothing
end

# Screen-aligned annotation pinned to a world-frame point. `offset` is in pixels
# so the label keeps a constant gap from the mesh regardless of the perspective,
# and every label carries the same slight upward tilt.
#
# The labels have to sit on top of the translucent gates. `overdraw` is not enough:
# it only lifts the text above the *opaque* pass, and the gates are composited over
# that afterwards, which washes the glyphs out. Joining the transparent pass with a
# depth shift hard toward the camera instead makes the text win the blend.
const LABEL_TILT = deg2rad(15)
const PAD        = (10, 10)     # pixel gap from the vehicle to the corner of the label
# translucent white halo around the glyphs, so a label crossing a gate frame or the
# flight path still reads as a separate layer
const HALO       = RGBAf(1, 1, 1, 0.8)
const HALO_W     = 2.5
const MU_ABOVE   = Set([2, 3])  # gates whose μ label sits above its quadrotor, not below

function annotate!(ax, pos, txt; color, fontsize=17, offset=PAD, align=(:left, :center))
    text!(ax, Point3f(pos...); text=txt, space=:data, align=align, offset=offset,
          fontsize=fontsize, color=color, rotation=LABEL_TILT,
          strokecolor=HALO, strokewidth=HALO_W,
          transparency=true, depth_shift=-1.0f0)
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

fig = let
    lims = (xmin, xmax, ymin, ymax, zmin, zmax)

    fig = Figure(size=(900, 700), figure_padding=0, backgroundcolor=:white)
    ax  = Axis3(fig[1, 1], aspect=:data, perspectiveness=0.8, limits=lims)
    hidedecorations!(ax); hidespines!(ax)

    # brighten the scene: a strong ambient keeps the light gate gray flat, while
    # the default camera-following directional light still gives the quadrotor
    # meshes gentle 3-D shading.
    set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

    # gates — uniform light blue-gray scenery (frame + legs + number)
    for i in 1:ngates
        plot_gate!(ax; pos=Point3f(Float32.(p.gate_positions[i])...),
                   R=p.gate_orientations[i], num=i)
    end

    # flown path
    lines!(ax, Point3f.(r_traj); color=:black, joinstyle=:round, linecap=:round,
           linewidth=1.2, alpha=0.5)

    # intermediate quadrotors, see-through so they read as background
    for k in shown_ks
        plot_quadrotor!(ax; p=Point3f(r_traj[k]...), q=q_traj[k], v=v_traj[k],
                        color=QUAD_COLOR, transparency=true)
    end

    # quadrotors passing through a gate (μ > 0), in red and labelled with the
    # progress multiplier that fired there.
    # μ hangs off its quadrotor's right, pinned by the near corner of the label and
    # padded clear of the rotors: below for most gates, above for the ones listed in
    # MU_ABOVE, whose vehicles have something in the way underneath. Several gates
    # cleared at one knot stack away from the vehicle.
    for (k, gates) in pass_knot
        plot_quadrotor!(ax; p=Point3f(r_traj[k]...), q=q_traj[k], v=v_traj[k], color=PASS_COLOR)
        for (n, j) in enumerate(gates)
            val = μ[k, j]
            vs  = isapprox(val, round(val); atol=5e-3) ? string(round(Int, val)) :
                  string(round(val, digits=2))
            up = j in MU_ABOVE
            annotate!(ax, r_traj[k], latexstring("\\mu^{($j)} = $vs");
                      color=LABEL_RED,
                      align=(:left, up ? :bottom : :top),
                      offset=(PAD[1], (up ? 1 : -1) * (PAD[2] + 16 * (n - 1))))
        end
    end

    # γ rides up and to the right instead: its bottom-left corner sits on the
    # neutral quadrotor, same padding.
    for k in shown_ks
        annotate!(ax, r_traj[k], latexstring("\\gamma = (" * join(γ_int(k), ",") * ")");
                  color=LABEL_DARK, align=(:left, :bottom), offset=PAD)
    end

    # ground grid (uniform grid-line weight across the 3 plots: lw=1.5 @ px_per_unit=6)
    xy_grid!(ax; xlim=(xmin, xmax), ylim=(ymin, ymax), z=FLOORZ, step=1.0,
             color=(:black, 0.15), lw=1.5)

    # press "p" to print the current camera angles (handy for tuning the save view)
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press && event.key == Keyboard.p
            @show ax.azimuth[]
            @show ax.elevation[]
        end
    end

    # front-on: the eye sits straight behind the start looking down the +y course
    # (azimuth = -π/2), lifted well above the gates so the loop of the flight path
    # opens up (elevation ≈ 40° above the horizon).
    ax.azimuth[]   = -π / 2
    ax.elevation[] = 0.70

    fig
end

const OUTPATH = joinpath(@__DIR__, "png_renders", "quadrotor_gates_3d_progress_glmakie.png")
mkpath(dirname(OUTPATH))
save(OUTPATH, fig, px_per_unit=6)
