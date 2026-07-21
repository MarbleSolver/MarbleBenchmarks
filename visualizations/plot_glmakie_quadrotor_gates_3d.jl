using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie.GeometryBasics: Cylinder

# ===========================================================================
# Quadrotor-gates trajectory — GLMakie render using the .obj meshes in
# visualizations/assets/. Loads the latest Marble solution from
# benchmarks/robotics/results/solutions/ (same data the TikZ perspective plot
# uses) and drops a shaded quadrotor mesh at knots along the flight path.
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

# knots at which each gate is cleared: μ[k, j] spikes when gate j is passed.
const MU_TOL = 1e-3
gate_knot = [(k, argmax(μ[k, :])) for k in 1:N if maximum(μ[k, :]) > MU_TOL]

# distinct per-gate palette (shared with plot_tikz_quadrotor_gates_3d_persp.jl)
const GATE_COLORS = ["#2C6FB0", "#E6A817", "#2E9E52", "#CB4154"]

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

# shaded material with a restrained specular so the bright gate colours don't
# blow out to white. Scene brightness is handled by the lights (see below).
const LIT = (; shading=true, specular=0.1f0, shininess=12.0f0)

function plot_quadrotor!(ax; p, q, v, color=:lightsteelblue3, s=0.075)
    quad_plot = mesh!(ax, quad; color=color, alpha=1.0, transparency=false, LIT...)

    GLMakie.scale!(quad_plot, (s, s, s))
    GLMakie.translate!(quad_plot, p...)

    # yaw the heading toward the velocity, keeping the solved bank (attitude q).
    yaw = norm(v) > 1e-6 ? atan(v[2], v[1]) : 0.0
    qyaw = [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]   # RotZ(yaw), scalar-first
    GLMakie.rotate!(quad_plot, makie_quat(qmul(qyaw, q)))

    return quad_plot
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

    # brighten the scene: a strong ambient keeps the bright gate colours
    # saturated, while the default camera-following directional light still
    # gives the meshes gentle 3-D shading.
    set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

    # gates — each in its own colour (frame + legs + number)
    for i in 1:ngates
        plot_gate!(ax; pos=Point3f(Float32.(p.gate_positions[i])...),
                   R=p.gate_orientations[i], num=i, color=GATE_COLORS[i], textc=GATE_COLORS[i])
    end

    # quadrotors: neutral along the path, highlighted in the gate's colour at
    # the knot where that gate is cleared.
    # light slate — reads as a solid step under the strong ambient light (a
    # near-white like aliceblue would blow out to invisible).
    c = RGBf(0.68, 0.73, 0.80)
    gate_ks = Set(k for (k, _) in gate_knot)
    for i in [1:5:N-5; N]
        i in gate_ks && continue
        plot_quadrotor!(ax; p=Point3f(r_traj[i]...), q=q_traj[i], v=v_traj[i], color=c)
    end
    for (k, j) in gate_knot
        plot_quadrotor!(ax; p=Point3f(r_traj[k]...), q=q_traj[k], v=v_traj[k], color=GATE_COLORS[j])
    end

    # flown path
    lines!(ax, Point3f.(r_traj); color=:black, joinstyle=:round, linecap=:round,
           linewidth=1.2, alpha=0.8)

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

    # low, looking-forward view matching the TikZ perspective camera
    # (eye behind the start, slightly elevated → ~21° above the horizon).
    ax.azimuth[]   = 4.428
    ax.elevation[] = 0.366

    fig
end

const OUTPATH = joinpath(@__DIR__, "png_renders", "quadrotor_gates_3d_glmakie.png")
mkpath(dirname(OUTPATH))
save(OUTPATH, fig, px_per_unit=6)

#= transparency + crop re-save disabled for now
# --- trim whitespace and make the background transparent --------------------
# Reload the rendered PNG, crop to the bounding box of the non-white content
# (leaving a small uniform margin), and key the near-white background out to
# alpha 0 so the saved PNG is transparent. A narrow alpha ramp just below the
# white cutoff softens anti-aliased edges so they don't leave a hard halo.
let img = load(OUTPATH)
    whiteness(c) = min(Float64(c.r), Float64(c.g), Float64(c.b))
    bgalpha(c)   = clamp((0.985 - whiteness(c)) / (0.985 - 0.965), 0.0, 1.0)
    isbg(c)      = bgalpha(c) <= 0.0
    H, W = size(img)
    rowhas = [any(!isbg(img[i, j]) for j in 1:W) for i in 1:H]
    colhas = [any(!isbg(img[i, j]) for i in 1:H) for j in 1:W]
    r0, r1 = findfirst(rowhas), findlast(rowhas)
    c0, c1 = findfirst(colhas), findlast(colhas)
    pad = round(Int, 0.02 * max(r1 - r0, c1 - c0))
    r0 = max(1, r0 - pad); r1 = min(H, r1 + pad)
    c0 = max(1, c0 - pad); c1 = min(W, c1 + pad)
    keyed = map(img[r0:r1, c0:c1]) do c
        RGBAf(Float64(c.r), Float64(c.g), Float64(c.b), bgalpha(c))
    end
    GLMakie.FileIO.save(OUTPATH, keyed)
end
=#
