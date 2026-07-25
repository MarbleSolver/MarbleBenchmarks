using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie.GeometryBasics
using GLMakie.GeometryBasics: Point3f, GLTriangleFace, normal_mesh
using GLMakie: Vec3f, RGBAf, RGBf, Rect3f
import GLMakie.Makie as MK      # AmbientLight / DirectionalLight for the bare Scene

GLMakie.activate!()
GLMakie.closeall() # close any open screen

# attitude helpers (reference quaternion is identity, so q = rptoq(δrp) is the
# body attitude quaternion directly and qtoQ(q) is the body→world rotation).
const rptoq = MarbleBenchmarks.rptoq
const qtoQ  = MarbleBenchmarks.qtoQ
const rocket_thrust_axis_body = MarbleBenchmarks.rocket_thrust_axis_body

# scalar-first quaternion → GLMakie.Quaternion (scalar-last / (x,y,z,w))
makie_quat(q) = GLMakie.Quaternion(q[2], q[3], q[4], q[1])

const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

prob = get_problem(:rocket_catch_3d)
inds, p = prob.inds, prob.params
N = p.N

z = deserialize(solfile)["rocket_catch_3d"]
X = z[inds[:x]]        # N × 14  reduced state [p(3); ϕ(3); v(3); ω(3); δr; δp]

# per-knot world-frame CoM, body attitude, and gimbal angles (x̄[1:3] = 0)
P       = [p.x̄[1:3] .+ X[k, 1:3] for k in 1:N]  # CoM position
qs      = [rptoq(X[k, 4:6]) for k in 1:N]        # scalar-first attitude quaternions
δr_traj = [X[k, 13] for k in 1:N]                # gimbal roll
δp_traj = [X[k, 14] for k in 1:N]                # gimbal pitch

# commanded thrust per knot. The control is u = [thrust; δ̇r; δ̇p] and the physical
# thrust is the reference hover thrust ū[1] = m·g plus the solved deviation. There
# is no control at the terminal knot, so it reuses the last one.
U           = z[inds[:u]]                         # (N-1) × 3 controls
thrust_traj = [p.ū[1] + U[min(k, N - 1), 1] for k in 1:N]

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

# state at a continuous knot parameter s ∈ [1, N]: position + gimbal linearly,
# attitude by slerp.
function traj_at(s::Real)
    s = clamp(float(s), 1.0, float(N))
    k0 = floor(Int, s); k1 = min(k0 + 1, N); α = s - k0
    pos = (1 - α) .* P[k0] .+ α .* P[k1]
    q   = qslerp(qs[k0], qs[k1], α)
    δr  = (1 - α) * δr_traj[k0] + α * δr_traj[k1]
    δp  = (1 - α) * δp_traj[k0] + α * δp_traj[k1]
    return pos, q, δr, δp
end

# commanded thrust at a continuous knot parameter (linear between knots)
function thrust_at(s::Real)
    s = clamp(float(s), 1.0, float(N)); k0 = floor(Int, s); k1 = min(k0 + 1, N); α = s - k0
    return (1 - α) * thrust_traj[k0] + α * thrust_traj[k1]
end

# shaded material with a restrained specular (matches the still); scene
# brightness is handled by a strong ambient light set on the axis scene.
const LIT = (; shading = true, specular = 0.1f0, shininess = 12.0f0)

# booster mesh: load the STL and pre-transform it into a canonical body frame so
# its long axis is the body +z (nose), its CoM sits at the origin, and it is
# scaled to the problem's booster dimensions. (Identical to the still.)
const RSCALE  = 1.15  # widen the very slender true radius a touch for visibility
const VIS_LEN = 0.70  # cosmetic: shorten the visible booster body (< 1 = stockier)

raw     = load(joinpath(@__DIR__, "assets", "starshipbooster.stl"))
rawpts  = GeometryBasics.coordinates(raw)
rawfcs  = GeometryBasics.faces(raw)
ay0, ay1 = extrema(p2[2] for p2 in rawpts)                     # axis (y) extent
rmax    = maximum(max(abs(p2[1]), abs(p2[3])) for p2 in rawpts) # cross-section radius
scale_axis = VIS_LEN * p.booster_height / (ay1 - ay0)
scale_r    = RSCALE * p.booster_radius / rmax
const L_VIS = VIS_LEN * p.l             # CoM→engine-plane, shortened to match body
com_off    = scale_axis * ay0 + L_VIS   # CoM is l above the base along the axis
# map mesh (x, y, z) → body via RotX(+90°): (x, −z, y), a *proper* rotation
# (det +1) so triangle winding — and hence the recomputed outward normals — is
# preserved.
bodypts = [Point3f(pt[1] * scale_r, -pt[3] * scale_r, pt[2] * scale_axis - com_off)
           for pt in rawpts]
const BOOSTER = normal_mesh(GeometryBasics.Mesh(bodypts, rawfcs))

const PLUME_LEN         = 0.011   # engine-plume length at max throttle (short flare)
const PLUME_HALF_ANGLE  = 15.0    # engine-plume cone half-angle, degrees (narrower)

# the plume grows/shrinks with the commanded thrust: the trajectory's thrust range
# is mapped to the plume *height* over a wide range, so the flame is a long cone at
# max throttle and a short one as the booster throttles down into the catch. Only
# the height (length) tracks thrust — the radius stays proportional to it, so the
# cone's half-angle is constant (the flame just gets taller/shorter, same shape).
const THR_LO, THR_HI = extrema(thrust_traj)
const PLUME_FLARE = tand(PLUME_HALF_ANGLE)   # fixed r/len ⇒ constant cone half-angle
thr_frac(thr) = THR_HI > THR_LO ? clamp((thr - THR_LO) / (THR_HI - THR_LO), 0.0, 1.0) : 1.0
plume_len(thr) = PLUME_LEN * (0.80 + 1.25 * thr_frac(thr))   # 0.80×‥2.05× (min still visible)
plume_rad(thr) = plume_len(thr) * PLUME_FLARE               # scales with length ⇒ fixed angle

# a solid cone: apex at `apex`, flaring to a base circle of radius `r` a distance
# `len` along `dir`. used for the translucent engine plume.
function cone_mesh(apex, dir, len, r; n = 24)
    d = normalize(collect(dir))
    a = abs(d[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    u = normalize(cross(d, a)); v = cross(d, u)
    c = collect(apex) .+ len .* d
    ring = [Point3f((c .+ r .* (cos(t) .* u .+ sin(t) .* v))...)
            for t in range(0, 2π, length = n + 1)[1:end-1]]
    verts = vcat([Point3f(apex...)], ring, [Point3f(c...)])   # apex, ring, base centre
    fcs = GLTriangleFace[]
    for i in 1:n
        j = i == n ? 1 : i + 1
        push!(fcs, (1, i + 1, j + 1))       # side face
        push!(fcs, (n + 2, j + 1, i + 1))   # base cap
    end
    return normal_mesh(GeometryBasics.Mesh(verts, fcs))
end

# engine-plume base (engine plane) and downstream direction for a booster at
# world CoM `center`, attitude `q`, gimbal (δr, δp).
function plume_geom(center, q, δr, δp)
    Q    = qtoQ(q)
    zb   = Q * [0.0, 0.0, 1.0]
    base = center .- L_VIS .* zb                            # engine plane
    dir  = -normalize(Q * rocket_thrust_axis_body(δr, δp))  # plume direction
    return base, dir
end

# GeometryBasics prism helpers (identical to the still)
box_mesh(xlo, xhi, ylo, yhi, zlo, zhi) =
    Rect3f(Vec3f(xlo, ylo, zlo), Vec3f(xhi - xlo, yhi - ylo, zhi - zlo))

# a solid prism from 4 base corners (CCW) extruded upward by thickness τ in z
function prism_mesh(base, τ)
    top   = [Point3f(c[1], c[2], c[3] + τ) for c in base]
    bot   = [Point3f(c...) for c in base]
    verts = vcat(bot, top)                        # 1:4 bottom, 5:8 top
    fcs = GLTriangleFace[
        (1, 2, 3), (1, 3, 4),                     # bottom
        (5, 7, 6), (5, 8, 7),                     # top
        (1, 5, 6), (1, 6, 2),                     # side 1-2
        (2, 6, 7), (2, 7, 3),                     # side 2-3
        (3, 7, 8), (3, 8, 4),                     # side 3-4
        (4, 8, 5), (4, 5, 1),                     # side 4-1
    ]
    return normal_mesh(GeometryBasics.Mesh(verts, fcs))
end

# a flat quad (two triangles) from 4 corners — the translucent trigger cap
function quad_mesh(a, b, c, d)
    verts = Point3f[a, b, c, d]
    fcs   = GLTriangleFace[(1, 2, 3), (1, 3, 4)]
    return normal_mesh(GeometryBasics.Mesh(verts, fcs))
end

# tower + chopstick arms (from the rocket_catch_3d params, mirroring the still)
const FLOORZ  = 0.0
const TOWER_H = p.tower_height
const CATCH_Z = p.catch_altitude
const aw      = p.arm_half_width
const slope   = tan(p.arm_opening_angle)

# tower body: rectangular prism, front face at y = 0 (the corridor mouth)
const TXW  = 0.0078      # tower half-width in x
const TDEP = 0.0150      # tower depth in y (front face at y = 0)

# splayed chopstick arms: a trapezoidal plate on each side (±x), reaching forward
# (+y) out of the tower face at the catch altitude, angled slightly up.
const ARM_LEN  = 0.040
const ARM_WR   = 0.0075
const ARM_WT   = 0.0055
const ARM_RISE = 0.002
const ARM_THK  = 0.0065
const ARM_YBK  = -TDEP

# corners of an arm (side s = ±1) for a given horizontal opening angle `ang`,
# spanning its full length y ∈ [ARM_YBK, ARM_LEN]. Varying `ang` rotates the
# plate about the tower-face hinge line (y = 0, x = s·aw) — the chopstick swing.
function arm_base(s, ang)
    tanA   = s * tan(ang)
    xin(y) = s * aw + tanA * y
    wof(y) = ARM_WR + (ARM_WT - ARM_WR) * (y - ARM_YBK) / (ARM_LEN - ARM_YBK)
    zof(y) = CATCH_Z + ARM_RISE * (y - ARM_YBK) / (ARM_LEN - ARM_YBK)
    ya, yb = ARM_YBK, ARM_LEN
    [[xin(ya), ya, zof(ya)], [xin(ya) + s * wof(ya), ya, zof(ya)],
     [xin(yb) + s * wof(yb), yb, zof(yb)], [xin(yb), yb, zof(yb)]]
end

# arm opening angle as a function of the booster's altitude z: fully splayed
# above Z_OPEN, easing closed (arms rotate inward) by the catch altitude.
const ARM_ANG_OPEN   = 0.62      # splayed (matches the still's static arms)
const ARM_ANG_CLOSED = -0.05     # converged to hug the booster at the catch
const Z_OPEN  = 0.135            # start closing once the booster drops below here
const Z_CLOSE = CATCH_Z          # fully closed at the catch altitude
function arm_angle(zc)
    c = clamp((Z_OPEN - zc) / (Z_OPEN - Z_CLOSE), 0.0, 1.0)
    c = c * c * (3 - 2c)         # smoothstep ease
    return ARM_ANG_OPEN + c * (ARM_ANG_CLOSED - ARM_ANG_OPEN)
end

# keep-in corridor {y >= 0, |x| <= aw + slope·y}, extruded z ∈ [0, z_trigger].
const YMAX  = 0.120
const ZTRIG = p.trigger_altitude
hwid(y)     = aw + slope * y

funnel_bot = [Point3f( hwid(0.0),  0.0,  FLOORZ), Point3f(-hwid(0.0),  0.0,  FLOORZ),
              Point3f(-hwid(YMAX), YMAX, FLOORZ), Point3f( hwid(YMAX), YMAX, FLOORZ)]
funnel_top = [Point3f(c[1], c[2], ZTRIG) for c in funnel_bot]

z_hi = maximum(pt[3] for pt in P) + VIS_LEN * p.booster_height / 2 + 0.03
lims = (-0.05, 0.05, -0.02, 0.13, 0.0, z_hi)

# simple ground grid at height z
function xy_grid!(ax; xlim, ylim, z, step = 0.02, color = (:black, 0.15), lw = 1)
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

# palette (shared with the still)
const C_TOWER  = RGBf(0.68, 0.73, 0.80)      # light slate (shared scene colour)
const C_ARM    = RGBf(0.55, 0.60, 0.67)      # slightly darker slate, same family
const C_FUNNEL = RGBf(0x2C / 255, 0x6F / 255, 0xB0 / 255)  # blue
const C_TRIG   = RGBf(0xE6 / 255, 0xA8 / 255, 0x17 / 255)  # gold

# time gradient — the same cool→warm ramp the still strobes across its knots
const ROCKET_GRAD = cgrad([RGBf(0x2C / 255, 0x6F / 255, 0xB0 / 255),   # blue
                           RGBf(0x7A / 255, 0x4F / 255, 0xB5 / 255),   # violet
                           RGBf(0xC1 / 255, 0x3C / 255, 0x9A / 255),   # magenta
                           RGBf(0xE6 / 255, 0x7E / 255, 0x22 / 255),   # orange
                           RGBf(0xCB / 255, 0x41 / 255, 0x54 / 255)])  # red (catch)
tcolor(frac) = ROCKET_GRAD[clamp(float(frac), 0.0, 1.0)]

const FPS     = 60          # video frame rate (high, for smooth motion)
const SLOWMO  = 0.8         # motion runs this × real time (1.0 = the 15 s solution)
const START_S = 0.6         # short hold on the initial pose
const END_S   = 1.4         # hold on the final catch pose

frames_per_knot = max(1, round(Int, SLOWMO * p.Δt * FPS))

# frame schedule: continuous knot parameters s, bracketed by start/end holds
s_frames = Float64[]
append!(s_frames, fill(1.0, round(Int, START_S * FPS)))
for k in 2:N
    for f in 1:frames_per_knot
        push!(s_frames, (k - 1) + f / frames_per_knot)   # advance up to exactly k
    end
end
append!(s_frames, fill(float(N), round(Int, END_S * FPS)))
const n_frames = length(s_frames)

# A bare Scene (no Axis3) driven by a real cam3d! camera, exactly like
# animate_glmakie_quadrotor_gates_3d.jl: the camera framing is decoupled from any
# data box, so the tower, full descent trail, corridor and trigger cap are never
# clipped as the view zooms/orbits (an Axis3's shrinking `limits` box clips them).
# Lighting is set explicitly since a bare Scene has no axis to carry it: strong
# ambient keeps the colours saturated, one directional light gives 3-D shading.
lights = [MK.AmbientLight(RGBf(0.65, 0.65, 0.65)),
          MK.DirectionalLight(RGBf(0.85, 0.85, 0.85), Vec3f(-0.4, -0.5, -0.9))]
scene = Scene(size = (820, 1000), backgroundcolor = :white, lights = lights)

# real 3-D camera with static near/far planes (small scene scale → small near),
# so framing is decoupled from the scene extent and nothing is clipped as it flies.
cam3d!(scene; fov = 45f0, near = 0.003f0, far = 20f0, clipping_mode = :static)

# ground grid
xy_grid!(scene; xlim = (lims[1], lims[2]), ylim = (lims[3], lims[4]), z = FLOORZ,
         step = 0.02, color = (:black, 0.15), lw = 1.5)

# tower body (static)
mesh!(scene, box_mesh(-TXW, TXW, -TDEP, 0.0, FLOORZ, TOWER_H); color = C_TOWER, LIT...)

# two chopstick arms — observable meshes rebuilt as the opening angle closes
arm_meshes = [Observable(prism_mesh(arm_base(s, ARM_ANG_OPEN), ARM_THK)) for s in (-1.0, 1.0)]
for am in arm_meshes
    mesh!(scene, am; color = C_ARM, LIT...)
end

# keep-in corridor: outline only (blue dashed edges + bottom loop)
for i in 1:4                               # vertical corridor edges
    lines!(scene, [funnel_bot[i], funnel_top[i]]; color = (C_FUNNEL, 0.9),
           linewidth = 1.4, linestyle = :dash)
end
lines!(scene, vcat(funnel_bot, [funnel_bot[1]]); color = (C_FUNNEL, 0.9),
       linewidth = 1.2, linestyle = :dash)

# trigger-altitude top cap (orange, translucent)
mesh!(scene, quad_mesh(funnel_top...); color = RGBAf(C_TRIG.r, C_TRIG.g, C_TRIG.b, 0.35),
      transparency = true, shading = false)
lines!(scene, vcat(funnel_top, [funnel_top[1]]); color = C_TRIG, linewidth = 2.0)

# growing CoM descent trail (black), revealed up to the current frame
trail = Observable(Point3f[Point3f(P[1]...)])
lines!(scene, trail; color = :black, joinstyle = :round, linecap = :round,
       linewidth = 1.2, alpha = 0.8)

# the single moving booster (fixed geometry; translate/rotate/colour update)
pos0, q0, δr0, δp0 = traj_at(1.0)
bcolor  = Observable{Any}(tcolor(0.0))
booster = mesh!(scene, BOOSTER; color = bcolor, LIT...)
GLMakie.rotate!(booster, makie_quat(q0))
GLMakie.translate!(booster, pos0...)

# gimballed engine plume cone + dotted orientation stub (observable geometry)
base0, dir0 = plume_geom(pos0, q0, δr0, δp0)
plen0 = plume_len(thrust_at(1.0)); prad0 = plume_rad(thrust_at(1.0))
plume_mesh_o = Observable(cone_mesh(base0, dir0, plen0, prad0))
plume_line_o = Observable(Point3f[Point3f(base0...),
                                  Point3f((base0 .+ 1.28 .* plen0 .* dir0)...)])
mesh!(scene, plume_mesh_o; color = RGBAf(1.0, 0.62, 0.12, 0.45),
      transparency = true, shading = false)
lines!(scene, plume_line_o; color = :black, linewidth = 0.6, linestyle = :dot)

function set_frame!(s)
    pos, q, δr, δp = traj_at(s)

    # booster body
    GLMakie.rotate!(booster, makie_quat(q))
    GLMakie.translate!(booster, pos...)
    bcolor[] = tcolor((s - 1) / (N - 1))

    # gimballed plume, sized by the commanded thrust
    base, dir = plume_geom(pos, q, δr, δp)
    plen = plume_len(thrust_at(s)); prad = plume_rad(thrust_at(s))
    plume_mesh_o[] = cone_mesh(base, dir, plen, prad)
    plume_line_o[] = Point3f[Point3f(base...),
                             Point3f((base .+ 1.28 .* plen .* dir)...)]

    # chopstick arms close inward as the booster descends into them
    ang = arm_angle(pos[3])
    for (si, sgn) in enumerate((-1.0, 1.0))
        arm_meshes[si][] = prism_mesh(arm_base(sgn, ang), ARM_THK)
    end

    # growing CoM trail up to the current position
    kf  = floor(Int, s)
    pts = Point3f[Point3f(P[m]...) for m in 1:kf]
    push!(pts, Point3f(pos...))
    trail[] = pts
    return pos
end

# Same mechanism as the quadrotor chase cam: a real cam3d! camera driven each
# frame by update_cam!(scene, eye, lookat, up), so framing is decoupled from any
# data box and nothing clips. The move is a keyframed dolly-orbit between two
# poses, eased over the approach window (knots CAM_S0 → CAM_S1):
#   • Start: a moderately-high, fairly-close look UP at the whole descent.
#   • End:   orbit counter-clockwise (turn left) and dolly in toward a 3/4 front
#            view of the catch, looking at the booster as it settles in.
# eye = lookat + dist · (cos el cos az, cos el sin az, sin el); lookat, dist, az,
# el are each lerped from their start value to their end value by the eased ct.
# az increases (CCW) from a +x-side view toward a +x/+y (front-right) eye; dist
# shrinks (zoom in); lookat pans from mid-descent down toward the catch.
#
# The END pose is the one that sat ~8 s into the previous cut (the old keyframes
# lerped at ct≈0.583): a 3/4 front view, tipped slightly up, moderately zoomed —
# the move now settles there and holds, rather than pushing on to a tight head-on.
const CAM_LA0   = Point3f(0.0, 0.055, 0.32)   # look at mid-descent (frames whole scene)
const CAM_DIST0 = 0.92                         # fairly close (raised + nearer than before)
const CAM_AZ0   = 5.678718749999997           # start azimuth (eye on the +x side)
const CAM_EL0   = -0.45                         # raised: less extreme look-up than before
const CAM_LA1   = Point3f(0.0, 0.0346, 0.1787) # 8 s look-at (partway down toward the catch)
const CAM_DIST1 = 0.657                         # 8 s distance (moderately zoomed)
const CAM_AZ1   = 6.857                         # 8 s azimuth (3/4 front, front-right eye)
const CAM_EL1   = -0.1892                       # 8 s elevation (tipped slightly up)
const CAM_UP    = Vec3f(0, 0, 1)
const CAM_S0    = 10.0                          # begin moving earlier in the descent
const CAM_S1    = 50.0                          # settled into the 8 s catch view by here

smoothstep(t) = (u = clamp(float(t), 0.0, 1.0); u * u * (3 - 2u))
cam_ct(s) = smoothstep((s - CAM_S0) / (CAM_S1 - CAM_S0))
lerp(a, b, t) = a .+ t .* (b .- a)
cam_dir(az, el) = Vec3f(cos(el) * cos(az), cos(el) * sin(az), sin(el))

cam_lookat(ct) = Point3f(lerp(CAM_LA0, CAM_LA1, ct)...)
function cam_eye(ct)
    la  = cam_lookat(ct)
    az  = CAM_AZ0   + ct * (CAM_AZ1   - CAM_AZ0)
    el  = CAM_EL0   + ct * (CAM_EL1   - CAM_EL0)
    d   = CAM_DIST0 + ct * (CAM_DIST1 - CAM_DIST0)
    return la + d * cam_dir(az, el)
end
set_camera!(ct) = update_cam!(scene, cam_eye(ct), cam_lookat(ct), CAM_UP)

set_frame!(1.0)
set_camera!(0.0)

# probe mode: render a few stills across the camera move (for tuning), then exit
if get(ENV, "RC_PROBE", "") != ""
    probedir = get(ENV, "RC_PROBE_DIR", @__DIR__)
    for (tag, s) in [("wide", 8.0), ("mid1", 30.0), ("mid2", 42.0), ("close", 54.0), ("end", 60.0)]
        set_frame!(s); set_camera!(cam_ct(s))
        save(joinpath(probedir, "rc_probe_$tag.png"), scene; px_per_unit = 2)
    end
    GLMakie.closeall(); exit()
end

const OUTPATH = joinpath(@__DIR__, "animations", "rocket_catch_3d_glmakie.mp4")
mkpath(dirname(OUTPATH))

record(scene, OUTPATH, 1:n_frames; framerate = FPS) do i
    s = s_frames[i]
    set_frame!(s)
    set_camera!(cam_ct(s))
end

@info "saved animation" OUTPATH n_frames FPS
