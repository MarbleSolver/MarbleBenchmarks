using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.GeometryBasics: Rect3f, Sphere, Cylinder
using GLMakie: Vec3f, Point3f, RGBAf

# ===========================================================================
# 2-D hopper trajectory — GLMakie render.
#
# Loads the latest Marble solution of hopper_2d from
# benchmarks/robotics/results/solutions/ (the same data the TikZ side-view uses)
# and draws the staircase terrain as shaded boxes, then drops a two-mass hopper
# (foot sphere + head sphere joined by a prismatic-leg cylinder) at knots along
# the motion, coloured by a time gradient, with the foot path overlaid.
#
# The scene is a side view: the GLMakie axes are (depth x ≈ 0, horizontal y,
# height z), so the model's (foot_x, foot_z) maps to (y, z).
# ===========================================================================

GLMakie.activate!()
GLMakie.closeall() # close any open screen

# --- load the latest Marble solution ---------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

# shaded material with a restrained specular (matches plot_glmakie_quadrotor_gates_3d.jl);
# scene brightness is handled by a strong ambient light set on the axis scene below.
const LIT = (; shading = true, specular = 0.1f0, shininess = 12.0f0)

problem = get_problem(:hopper_2d)
inds, p = problem.inds, problem.params
N = p.N

z = deserialize(solfile)["hopper_2d"]
X = z[inds[:x]]        # (N+1) × 8  state [foot_x, foot_z, head_x, head_z, ...]

# per-knot state vectors (each row of X)
x_traj = [X[k, :] for k in 1:size(X, 1)]

# helper: MeshCat-style box from HyperRectangle(min, size)
function box!(ax, min::NTuple{3,Real}, size::NTuple{3,Real};
              color = RGBAf(0, 0.5, 1, 0.7))
    r = Rect3f(Vec3f(Float32(min[1]), Float32(min[2]), Float32(min[3])),
               Vec3f(Float32(size[1]), Float32(size[2]), Float32(size[3])))
    mesh!(ax, r; color = color, LIT...)
    return r
end

# draw the two-mass hopper at state x = [foot_x, foot_z, head_x, head_z, ...]
function plot_hopper!(ax, x;
    h::Real = 0.25,
    r::Real = 0.05,
    body_r::Real = 0.02,
    foot_color = :black,
    head_color = :black,
    body_color = (:gray, 0.9),
)
    # side view: (depth = 0, horizontal = *_x, height = *_z)
    pfoot = Point3f(0, x[1], x[2] + r)
    phead = Point3f(0, x[3], x[4] + h + r)

    foot = mesh!(ax, Sphere(pfoot, Float32(r)); color = foot_color, LIT...)
    head = mesh!(ax, Sphere(phead, Float32(1.5 * r)); color = head_color, LIT...)

    # body as a cylinder from foot to head
    body = mesh!(ax, Cylinder(pfoot, phead, Float32(body_r)); color = body_color, LIT...)

    return (foot = foot, head = head, body = body, pfoot = pfoot, phead = phead)
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

    h = 0.25
    r = 0.03
    body_r = 0.01
    Nk = length(x_traj)

fig = let
    # Ax limits (depth x, horizontal y, height z), sized to the data so nothing is
    # clipped: horizontal spans foot/head x, height reaches the raised head sphere.
    y_lo = minimum(min(x[1], x[3]) for x in x_traj)
    y_hi = maximum(max(x[1], x[3]) for x in x_traj)
    z_hi = maximum(x[4] + h + 1.5r for x in x_traj)   # top of the head sphere
    xmin, xmax = -0.5, 0.5
    ymin, ymax = y_lo - 0.3, y_hi + 0.3
    zmin, zmax = 0.0, z_hi + 0.1
    lims = (xmin, xmax, ymin, ymax, zmin, zmax)

    fig = Figure(size = (900, 700))
    ax  = Axis3(fig[1, 1], aspect = :data, perspectiveness = 0.8, limits = lims)
    hidedecorations!(ax); hidespines!(ax)

    # brighten the scene: a strong ambient keeps the mesh colours saturated while
    # the default camera-following directional light gives gentle 3-D shading
    # (matches plot_glmakie_quadrotor_gates_3d.jl).
    set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

    # light slate — reads as a solid step under the strong ambient light (a near-white
    # like aliceblue would blow out to invisible).
    c = RGBf(0.68, 0.73, 0.80)

    # staircase terrain: three rising steps (min, size) in (depth, horizontal, height)
    box!(ax, (-0.1, 0.5,  0.0), (0.2, 0.25, 0.1); color = c)   # step1
    box!(ax, (-0.1, 0.75, 0.0), (0.2, 0.25, 0.2); color = c)   # step2
    box!(ax, (-0.1, 1.0,  0.0), (0.2, 1.0,  0.3); color = c)   # step3

    tcolors = cgrad(:diverging_rainbow_bgymr_45_85_c67_n256, Nk; categorical = true).colors

    for i in [3:4:Nk-3; Nk]
        x = x_traj[i]
        ci = tcolors[i]
        plot_hopper!(ax, x; h = h, r = r, body_r = body_r,
                     foot_color = ci, head_color = ci, body_color = ci)
    end

    # foot path (depth = 0, horizontal = foot_x, height = foot_z)
    r_traj = [Point3f(0, x[1], x[2]) for x in x_traj]
    lines!(ax, r_traj; color = :black, joinstyle = :round, linecap = :round,
           linewidth = 1.2, alpha = 0.8)

    xy_grid!(ax; xlim = (xmin, xmax), ylim = (ymin, ymax), z = 0.0,
             step = 0.25, color = (:black, 0.15), lw = 1)

    # press "p" to print the current camera angles (handy for tuning the save view)
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press && event.key == Keyboard.p
            @show ax.azimuth[]
            @show ax.elevation[]
        end
    end

    # good az/el for saving
    ax.azimuth[]   = -0.9323046875000006
    ax.elevation[] = 0.37051158169872134
    fig
end

const OUTPATH = joinpath(@__DIR__, "hopper_2d_glmakie.png")
save(OUTPATH, fig; px_per_unit = 6)

# --- trim the surrounding whitespace ---------------------------------------
# reload the rendered PNG and crop to the bounding box of everything that isn't
# the (near-white) background, leaving a small uniform margin.
let img = GLMakie.FileIO.load(OUTPATH)
    isbg(c) = Float64(c.r) > 0.985 && Float64(c.g) > 0.985 && Float64(c.b) > 0.985
    H, W = size(img)
    rowhas = [any(!isbg(img[i, j]) for j in 1:W) for i in 1:H]
    colhas = [any(!isbg(img[i, j]) for i in 1:H) for j in 1:W]
    r0, r1 = findfirst(rowhas), findlast(rowhas)
    c0, c1 = findfirst(colhas), findlast(colhas)
    pad = round(Int, 0.02 * max(r1 - r0, c1 - c0))
    r0 = max(1, r0 - pad); r1 = min(H, r1 + pad)
    c0 = max(1, c0 - pad); c1 = min(W, c1 + pad)
    GLMakie.FileIO.save(OUTPATH, img[r0:r1, c0:c1])
end
