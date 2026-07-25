using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.GeometryBasics: Rect3f, Sphere, Cylinder
using GLMakie: Vec3f, Point3f, RGBf, RGBAf

GLMakie.activate!()
GLMakie.closeall() # close any open screen

# By default use this specific saved solution; pass a path as ARGS[1] to override.
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results_old", "solutions")
const DEFAULT_SOL = joinpath(SOLDIR, "marble_2026-07-19_04-22-01.jls")

solfile = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_SOL

# shaded material with a restrained specular (matches the still); scene brightness
# is handled by a strong ambient light set on the axis scene below.
const LIT = (; shading = true, specular = 0.1f0, shininess = 12.0f0)

problem = get_problem(:hopper_2d)
inds, p = problem.inds, problem.params
N, dt = p.N, p.dt

z = deserialize(solfile)["hopper_2d"]
X = z[inds[:x]]        # (N+1) × 8  state [foot_x, foot_z, head_x, head_z, ...]

# per-knot state vectors (each row of X)
x_traj = [X[k, :] for k in 1:size(X, 1)]
Nk = length(x_traj)

# linearly interpolate the state at a continuous knot parameter s ∈ [1, Nk]
function interp_state(s::Real)
    s = clamp(float(s), 1.0, float(Nk))
    k0 = floor(Int, s)
    k1 = min(k0 + 1, Nk)
    α  = s - k0
    return (1 - α) .* x_traj[k0] .+ α .* x_traj[k1]
end

const HOP_H  = 0.25     # leg length offset used to draw the head above the foot
const HOP_R  = 0.03     # foot sphere radius
const BODY_R = 0.01     # leg cylinder radius

# helper: MeshCat-style box from HyperRectangle(min, size)
function box!(ax, min::NTuple{3,Real}, size::NTuple{3,Real};
              color = RGBAf(0, 0.5, 1, 0.7))
    r = Rect3f(Vec3f(Float32(min[1]), Float32(min[2]), Float32(min[3])),
               Vec3f(Float32(size[1]), Float32(size[2]), Float32(size[3])))
    mesh!(ax, r; color = color, LIT...)
    return r
end

# simple ground grid at height z. Extents are snapped inward to whole multiples of
# `step` so the grid shows only complete cells (no partial cell at the edges).
function xy_grid!(ax; xlim, ylim, z, step = 0.25, color = (:black, 0.15), lw = 1)
    x0, x1 = ceil(xlim[1] / step) * step, floor(xlim[2] / step) * step
    y0, y1 = ceil(ylim[1] / step) * step, floor(ylim[2] / step) * step
    segs = Point3f[]
    for x in x0:step:x1
        push!(segs, Point3f(x, y0, z)); push!(segs, Point3f(x, y1, z))
    end
    for y in y0:step:y1
        push!(segs, Point3f(x0, y, z)); push!(segs, Point3f(x1, y, z))
    end
    linesegments!(ax, segs; color = color, linewidth = lw)
end

# side-view foot/head sphere centres for a state x = [foot_x, foot_z, head_x, head_z, ...]
foot_pt(x; r = HOP_R) = Point3f(0, x[1], x[2] + r)
head_pt(x; h = HOP_H, r = HOP_R) = Point3f(0, x[3], x[4] + h + r)

# time gradient (continuous) — the same colormap the still strobes across the knots
const TGRAD = cgrad(:diverging_rainbow_bgymr_45_85_c67_n256)
tcolor(frac) = TGRAD[clamp(float(frac), 0.0, 1.0)]

const FPS     = 60          # video frame rate (high, for smooth motion)
const SLOWMO  = 1.5         # video runs this much longer than real time (<1 speeds up)
const HOLD_S  = 1.0         # freeze on the final pose for this many seconds

n_motion = max(2, round(Int, N * dt * SLOWMO * FPS))
n_hold   = round(Int, HOLD_S * FPS)
n_frames = n_motion + n_hold

# knot parameter s and time fraction for each frame
sfor(i) = i <= n_motion ? 1 + (i - 1) / (n_motion - 1) * (Nk - 1) : float(Nk)
fracfor(i) = (sfor(i) - 1) / (Nk - 1)

fig = let h = HOP_H, r = HOP_R, body_r = BODY_R
    # Ax limits (depth x, horizontal y, height z), sized to the data so nothing is
    # clipped — identical framing to the still so the animation matches it.
    y_lo = minimum(min(x[1], x[3]) for x in x_traj)
    y_hi = maximum(max(x[1], x[3]) for x in x_traj)
    z_hi = maximum(x[4] + h + 1.5r for x in x_traj)   # top of the head sphere
    xmin, xmax = -0.5, 0.5
    ymin, ymax = y_lo - 0.3, y_hi + 0.3
    zmin, zmax = 0.0, z_hi + 0.1
    lims = (xmin, xmax, ymin, ymax, zmin, zmax)

    fig = Figure(size = (900, 700), figure_padding = 0, backgroundcolor = :white)
    ax  = Axis3(fig[1, 1], aspect = :data, perspectiveness = 0.8, limits = lims)
    hidedecorations!(ax); hidespines!(ax)

    # brighten the scene: a strong ambient keeps the mesh colours saturated while
    # the default camera-following directional light gives gentle 3-D shading.
    set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

    # light slate — reads as a solid step under the strong ambient light.
    c = RGBf(0.68, 0.73, 0.80)

    # staircase terrain: three rising steps (min, size) in (depth, horizontal, height)
    box!(ax, (-0.1, 0.5,  0.0), (0.2, 0.25, 0.1); color = c)   # step1
    box!(ax, (-0.1, 0.75, 0.0), (0.2, 0.25, 0.2); color = c)   # step2
    box!(ax, (-0.1, 1.0,  0.0), (0.2, 1.0,  0.3); color = c)   # step3

    # ground grid (matches the still's weight)
    xy_grid!(ax; xlim = (xmin, xmax), ylim = (ymin, ymax), z = 0.0,
             step = 0.25, color = (:black, 0.15), lw = 1.5)

    footpath = [Point3f(0, interp_state(sfor(i))[1], interp_state(sfor(i))[2])
                for i in 1:n_motion]
    trail = Observable(Point3f[footpath[1]])
    lines!(ax, trail; color = :black, joinstyle = :round, linecap = :round,
           linewidth = 1.2, alpha = 0.8)

    x0    = interp_state(1.0)
    pfoot = Observable(foot_pt(x0; r = r))
    phead = Observable(head_pt(x0; h = h, r = r))
    hcol  = Observable(RGBAf(tcolor(0.0)))

    foot_sphere = lift(pf -> Sphere(pf, Float32(r)), pfoot)
    head_sphere = lift(ph -> Sphere(ph, Float32(1.5r)), phead)
    body_cyl    = lift((pf, ph) -> Cylinder(pf, ph, Float32(body_r)), pfoot, phead)

    mesh!(ax, foot_sphere; color = hcol, LIT...)
    mesh!(ax, head_sphere; color = hcol, LIT...)
    mesh!(ax, body_cyl;    color = hcol, LIT...)

    # good az/el for the render (same view as the still)
    ax.azimuth[]   = 5.428890008326985
    ax.elevation[] = 0.3483631441987243

    (fig, ax, pfoot, phead, hcol, trail, footpath, r, h)
end

fig, ax, pfoot, phead, hcol, trail, footpath, r, h = fig

const OUTPATH = joinpath(@__DIR__, "animations", "hopper_2d_glmakie.mp4")
mkpath(dirname(OUTPATH))

record(fig, OUTPATH, 1:n_frames; framerate = FPS) do i
    s = sfor(i)
    x = interp_state(s)
    pfoot[] = foot_pt(x; r = r)
    phead[] = head_pt(x; h = h, r = r)
    hcol[]  = RGBAf(tcolor(fracfor(i)))
    j = min(i, n_motion)
    trail[] = footpath[1:j]
end

@info "saved animation" OUTPATH n_frames FPS
