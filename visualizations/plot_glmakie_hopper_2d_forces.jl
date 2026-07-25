using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.GeometryBasics: Rect3f, Sphere, Cylinder
using GLMakie: Vec3f, Point3f, RGBf, RGBAf

GLMakie.activate!()
GLMakie.closeall() # close any open screen

const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

# shaded material with a restrained specular (matches plot_glmakie_hopper_2d.jl);
# scene brightness is handled by a strong ambient light set on the axis scene below.
const LIT = (; shading = true, specular = 0.1f0, shininess = 12.0f0)

problem = get_problem(:hopper_2d)
inds, p = problem.inds, problem.params
N = p.N

z = deserialize(solfile)["hopper_2d"]
X = z[inds[:x]]        # (N+1) × 8  state [foot_x, foot_z, head_x, head_z, ...]
F = z[inds[:f]]        # N × 2      contact force [tangential; normal] (scaled)

# per-knot state vectors (each row of X)
x_traj = [X[k, :] for k in 1:size(X, 1)]
Nk = length(x_traj)

# f[k, :] is the force in the dynamics update that lands on state k+1, so knot k+1
# is the one in contact. Forces are stored scaled by f_scale. The initial knot has no
# force variable of its own — it starts at rest on the floor, so it shares the force
# of the first interval and counts as a contact knot (never a gray flight one).
Fk = [F[max(k - 1, 1), :] * p.f_scale for k in 1:Nk]
const F_TOL = 2e-5 * p.f_scale
is_contact = [Fk[k][2] > F_TOL for k in 1:Nk]
stance = findall(is_contact)
fmag_max = maximum(norm(f) for f in Fk)

footx(k) = x_traj[k][1]

# The hopper barely advances during a stance phase, so drawing every stance knot
# piles a dozen hoppers on top of each other. Group each stance phase into clusters
# of knots that share (roughly) a foot position and keep the one carrying the
# largest force — the visible push-off of that contact.
const MIN_DX = 0.055     # minimum foot separation within a stance cluster [m]

function thin_stance(stance)
    isempty(stance) && return Int[]
    clusters, cur = Vector{Int}[], [stance[1]]
    for k in stance[2:end]
        if k - cur[end] <= 2 && abs(footx(k) - footx(cur[1])) < MIN_DX
            push!(cur, k)
        else
            push!(clusters, cur); cur = [k]
        end
    end
    push!(clusters, cur)
    return [c[argmax([norm(Fk[k]) for k in c])] for c in clusters]
end

stance_shown = thin_stance(stance)

# Flight strobe: every airborne knot of the trajectory.
flight_shown = findall(!, is_contact)

@info "contact knots" stance stance_shown fmag_max nflight = length(flight_shown)

const HOP_H  = 0.25     # leg length offset used to draw the head above the foot
const HOP_R  = 0.03     # foot sphere radius
const BODY_R = 0.01     # leg cylinder radius

const VIEW_AZ = 0.0     # camera azimuth: straight down the depth axis
const VIEW_EL = 0.11    # camera elevation [rad]: a few degrees above the horizon

const DEPTH_FLIGHT  = 0.15   # depth (x) at which the hoppers are drawn: in front of the
const DEPTH_STANCE  = 0.16   # terrain boxes (which span x ∈ [-0.1, 0.1]).
const DEPTH_ARROW   = 0.18

# Pulling a hopper toward the camera must not move it on screen, otherwise the feet
# would float above (or sink into) the terrain they touch. In an orthographic
# projection a shift along the view axis is invisible, so pair every depth offset with
# the matching rise in height.
lift(depth::Real) = depth * tan(VIEW_EL)
at_depth(depth::Real, y::Real, zz::Real) = Point3f(depth, y, zz + lift(depth))

const TERRAIN_COLOR = RGBf(0.68, 0.73, 0.80)      # light slate for the staircase boxes
# the flight strobe is see-through so it reads as background; the stance hoppers stay
# opaque in a light red that the bright force arrows sit on top of.
const FLIGHT_COLOR  = RGBAf(0.68, 0.73, 0.80, 0.40)   # airborne hoppers: terrain slate
const CONTACT_COLOR = RGBf(0.890, 0.604, 0.627)       # stance hoppers: light crimson
const ARROW_COLOR   = RGBf(0.898, 0.106, 0.145)   # bright red, opaque over the faded leg
const ARROW_LEN_MAX = 0.30                        # data units for the largest contact force

# helper: MeshCat-style box from HyperRectangle(min, size)
function box!(ax, min::NTuple{3, Real}, size::NTuple{3, Real};
              color = RGBAf(0, 0.5, 1, 0.7))
    r = Rect3f(Vec3f(Float32(min[1]), Float32(min[2]), Float32(min[3])),
               Vec3f(Float32(size[1]), Float32(size[2]), Float32(size[3])))
    mesh!(ax, r; color = color, LIT...)
    return r
end

# draw the two-mass hopper at state x = [foot_x, foot_z, head_x, head_z, ...]
function plot_hopper!(ax, x;
    h::Real = HOP_H,
    r::Real = HOP_R,
    body_r::Real = BODY_R,
    depth::Real = 0.0,
    foot_color = :black,
    head_color = :black,
    body_color = (:gray, 0.9),
    transparency = true,
)
    # side view: (depth, horizontal = *_x, height = *_z)
    pfoot = at_depth(depth, x[1], x[2] + r)
    phead = at_depth(depth, x[3], x[4] + h + r)

    foot = mesh!(ax, Sphere(pfoot, Float32(r)); color = foot_color, transparency, LIT...)
    head = mesh!(ax, Sphere(phead, Float32(1.5 * r)); color = head_color, transparency, LIT...)

    # body as a cylinder from foot to head
    body = mesh!(ax, Cylinder(pfoot, phead, Float32(body_r)); color = body_color,
                 transparency, LIT...)

    return (foot = foot, head = head, body = body, pfoot = pfoot, phead = phead)
end

# simple ground grid at height z. Extents are snapped inward to whole multiples of
# `step` so the grid shows only complete cells (no partial cell at the edges).
# Seen exactly side-on this collapses to a single ground line.
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

fig = let h = HOP_H, r = HOP_R
    # Ax limits (depth x, horizontal y, height z), sized to the data so nothing is
    # clipped: horizontal spans foot/head x, height reaches the raised head sphere.
    y_lo = minimum(min(x[1], x[3]) for x in x_traj)
    y_hi = maximum(max(x[1], x[3]) for x in x_traj)
    z_hi = maximum(x[4] + h + 1.5r for x in x_traj)   # top of the head sphere
    xmin, xmax = -0.5, 0.5
    ymin, ymax = y_lo - 0.3, y_hi + 0.3
    zmin, zmax = 0.0, max(z_hi, ARROW_LEN_MAX) + 0.06
    lims = (xmin, xmax, ymin, ymax, zmin, zmax)

    fig = Figure(size = (1200, 300), backgroundcolor = :white)
    # perspectiveness = 0 ⇒ orthographic, so the side-on view has no foreshortening
    ax  = Axis3(fig[1, 1], aspect = :data, perspectiveness = 0.0, limits = lims)
    hidedecorations!(ax); hidespines!(ax)

    # brighten the scene: a strong ambient keeps the mesh colours saturated while
    # the default camera-following directional light gives gentle 3-D shading.
    set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

    # light slate — reads as a solid step under the strong ambient light (a near-white
    # like aliceblue would blow out to invisible).
    c = TERRAIN_COLOR

    # staircase terrain: three rising steps (min, size) in (depth, horizontal, height)
    box!(ax, (-0.1, 0.5,  0.0), (0.2, 0.25, 0.1); color = c)   # step1
    box!(ax, (-0.1, 0.75, 0.0), (0.2, 0.25, 0.2); color = c)   # step2
    box!(ax, (-0.1, 1.0,  0.0), (0.2, 1.0,  0.3); color = c)   # step3

    # ground grid — a shallow band at this elevation, so it is kept narrow in depth and
    # faint enough to read as a floor rather than as a second subject.
    xy_grid!(ax; xlim = (-0.3, 0.3), ylim = (ymin, ymax), z = 0.0,
             step = 0.25, color = (:black, 0.10), lw = 1.2)

    # foot path (horizontal = foot_x, height = foot_z)
    r_traj = [at_depth(DEPTH_FLIGHT, x[1], x[2]) for x in x_traj]
    lines!(ax, r_traj; color = (:black, 0.55), joinstyle = :round, linecap = :round,
           linewidth = 1.2)

    # every airborne knot, in the same opaque slate as the terrain so the strobe reads
    # as background and only the stance knots carry colour. Drawn first so the red
    # stance hoppers land on top.
    for k in flight_shown
        plot_hopper!(ax, x_traj[k]; depth = DEPTH_FLIGHT,
                     foot_color = FLIGHT_COLOR, head_color = FLIGHT_COLOR,
                     body_color = FLIGHT_COLOR)
    end

    # stance hoppers, highlighted in red (opaque, so they cut cleanly out of the strobe)
    for k in stance_shown
        plot_hopper!(ax, x_traj[k]; depth = DEPTH_STANCE, transparency = false,
                     foot_color = CONTACT_COLOR, head_color = CONTACT_COLOR,
                     body_color = CONTACT_COLOR)
    end

    # 3-D contact-force arrows at the feet of the stance hoppers. Directions are the
    # real (unscaled) contact forces, uniformly scaled so the largest is ARROW_LEN_MAX.
    if !isempty(stance_shown)
        apos = [at_depth(DEPTH_ARROW, x_traj[k][1], x_traj[k][2] + r) for k in stance_shown]
        adir = [Vec3f(0, Fk[k][1], Fk[k][2]) for k in stance_shown]
        arrows3d!(ax, apos, adir;
                  lengthscale = ARROW_LEN_MAX / fmag_max,
                  markerscale = 1.0,          # radii/lengths below are in data units
                  shaftradius = 0.006, tipradius = 0.017, tiplength = 0.045,
                  minshaftlength = 0.0, maxshaftlength = Inf,
                  color = ARROW_COLOR)
    end

    # press "p" to print the current camera angles (handy for tuning the save view)
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press && event.key == Keyboard.p
            @show ax.azimuth[]
            @show ax.elevation[]
        end
    end

    # essentially side-on — camera on the +x axis looking down -x, horizontal y to the
    # right and height z up — but raised a few degrees so the scene is read slightly
    # from above and the step tops catch a sliver of light.
    ax.azimuth[]   = VIEW_AZ
    ax.elevation[] = VIEW_EL
    fig
end

# # Show an interactive GLMakie window and block so it stays open (and keyboard
# # events like "p" work). Running the script exits the process as soon as it
# # finishes, so without this wait the window would flash and close immediately.
# screen = display(fig)
# isinteractive() || wait(screen)

const OUTPATH = joinpath(@__DIR__, "png_renders", "hopper_2d_forces_glmakie.png")
mkpath(dirname(OUTPATH))
save(OUTPATH, fig; px_per_unit = 6)
