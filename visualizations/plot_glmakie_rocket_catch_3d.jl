using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie.GeometryBasics
using GLMakie.GeometryBasics: Point3f, GLTriangleFace, normal_mesh, Cylinder
using GLMakie: Vec3f, RGBAf, RGBf

# ===========================================================================
# Rocket-catch (Mechazilla) trajectory — 3-D GLMakie render, in the same style
# as plot_glmakie_quadrotor_gates_3d.jl / plot_glmakie_hopper_2d.jl and using
# the same Marble solution the TikZ perspective plot draws
# (plot_tikz_rocket_catch_3d_persp.jl, whose "high_left_r2" camera this view
# matches).
#
# Scene elements
#   • depth/time-shaded CoM descent trajectory
#   • scaled starshipbooster.stl mesh dropped at knots along the descent, each
#     at its solved attitude, with a short red engine-gimbal stub out of the base
#   • Mechazilla tower (rectangular prism) + two splayed chopstick catch arms
#     (extruded trapezoidal prisms) built from GeometryBasics primitives, placed
#     from the rocket_catch_3d params
#   • the translucent keep-in corridor {y >= 0, |x| <= aw + slope·y} extruded up
#     to the trigger altitude, whose translucent orange top face marks z_trigger
# ===========================================================================

GLMakie.activate!()
GLMakie.closeall() # close any open screen

# attitude helpers (reference quaternion is identity, so q = rptoq(δrp) is the
# body attitude quaternion directly and qtoQ(q) is the body→world rotation).
const rptoq = MarbleBenchmarks.rptoq
const qtoQ  = MarbleBenchmarks.qtoQ
const rocket_thrust_axis_body = MarbleBenchmarks.rocket_thrust_axis_body

# scalar-first quaternion → GLMakie.Quaternion (scalar-last / (x,y,z,w))
makie_quat(q) = GLMakie.Quaternion(q[2], q[3], q[4], q[1])

# --- load the latest Marble solution ---------------------------------------
const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

prob = get_problem(:rocket_catch_3d)
inds, p = prob.inds, prob.params
N = p.N

z = deserialize(solfile)["rocket_catch_3d"]
X = z[inds[:x]]        # N × 14  reduced state [p(3); ϕ(3); v(3); ω(3); δr; δp]

# world-frame CoM position and body attitude at each knot (x̄[1:3] = 0)
P = [p.x̄[1:3] .+ X[k, 1:3] for k in 1:N]
Qs = [qtoQ(rptoq(X[k, 4:6])) for k in 1:N]     # body→world rotation matrices
qs = [rptoq(X[k, 4:6]) for k in 1:N]           # scalar-first attitude quaternions

# shaded material with a restrained specular (matches the other GLMakie plots);
# scene brightness is handled by a strong ambient light set on the axis scene.
const LIT = (; shading = true, specular = 0.1f0, shininess = 12.0f0)

# ---------------------------------------------------------------------------
# booster mesh: load the STL and pre-transform it into a canonical body frame so
# its long axis is the body +z (nose), its CoM sits at the origin, and it is
# scaled to the problem's booster dimensions. The STL long axis is +y (base near
# y = 0, nose at y ≈ 1) with a ~0.135 cross-section radius.
# ---------------------------------------------------------------------------
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
# preserved. (A plain y↔z swap is a reflection that flips the normals inward and
# leaves the booster lit only by ambient, i.e. too dark.)
bodypts = [Point3f(pt[1] * scale_r, -pt[3] * scale_r, pt[2] * scale_axis - com_off)
           for pt in rawpts]
const BOOSTER = normal_mesh(GeometryBasics.Mesh(bodypts, rawfcs))

const RKT_H     = p.booster_height
const PLUME_LEN = 0.011      # engine-plume length (short flare out of the base)
const PLUME_R   = 0.0042     # engine-plume flare radius at the downstream end

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

# drop the booster mesh at world CoM `center`, attitude `q` (scalar-first), with
# a short translucent yellow/orange engine plume along the gimballed thrust axis
# out of the base.
function plot_rocket!(ax, center, q; color, δr = 0.0, δp = 0.0, engine = true)
    m = mesh!(ax, BOOSTER; color = color, LIT...)
    GLMakie.rotate!(m, makie_quat(q))
    GLMakie.translate!(m, center...)

    if engine
        Q    = qtoQ(q)
        zb   = Q * [0.0, 0.0, 1.0]
        base = center .- L_VIS .* zb                               # engine plane
        dir  = -normalize(Q * rocket_thrust_axis_body(δr, δp))     # plume direction
        mesh!(ax, cone_mesh(base, dir, PLUME_LEN, PLUME_R);
              color = RGBAf(1.0, 0.62, 0.12, 0.45), transparency = true, shading = false)
        # thin dotted orientation line: engine base → just past the plume cone's end
        line_end = base .+ 1.28 .* PLUME_LEN .* dir
        lines!(ax, [Point3f(base...), Point3f(line_end...)];
               color = :black, linewidth = 0.6, linestyle = :dot)
    end
    return m
end

# ---------------------------------------------------------------------------
# GeometryBasics prism helpers
# ---------------------------------------------------------------------------
# an axis-aligned rectangular prism from (xlo..xhi, ylo..yhi, zlo..zhi)
function box_mesh(xlo, xhi, ylo, yhi, zlo, zhi)
    r = Rect3f(Vec3f(xlo, ylo, zlo), Vec3f(xhi - xlo, yhi - ylo, zhi - zlo))
    return r
end

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

# a flat quad (two triangles) from 4 corners — used for the translucent walls
function quad_mesh(a, b, c, d)
    verts = Point3f[a, b, c, d]
    fcs   = GLTriangleFace[(1, 2, 3), (1, 3, 4)]
    return normal_mesh(GeometryBasics.Mesh(verts, fcs))
end

# ---------------------------------------------------------------------------
# tower + chopstick arms (from the rocket_catch_3d params, mirroring the TikZ)
# ---------------------------------------------------------------------------
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
const ARM_ANG  = 0.62
const ARM_WR   = 0.0075
const ARM_WT   = 0.0055
const ARM_RISE = 0.002
const ARM_THK  = 0.0065
const ARM_YBK  = -TDEP

# corners of an arm (side s = ±1) spanning its full length y ∈ [ARM_YBK, ARM_LEN]
function arm_base(s)
    tanA   = s * tan(ARM_ANG)
    xin(y) = s * aw + tanA * y
    wof(y) = ARM_WR + (ARM_WT - ARM_WR) * (y - ARM_YBK) / (ARM_LEN - ARM_YBK)
    zof(y) = CATCH_Z + ARM_RISE * (y - ARM_YBK) / (ARM_LEN - ARM_YBK)
    ya, yb = ARM_YBK, ARM_LEN
    [[xin(ya), ya, zof(ya)], [xin(ya) + s * wof(ya), ya, zof(ya)],
     [xin(yb) + s * wof(yb), yb, zof(yb)], [xin(yb), yb, zof(yb)]]
end

# ---------------------------------------------------------------------------
# keep-in corridor {y >= 0, |x| <= aw + slope·y}, extruded z ∈ [0, z_trigger].
# translucent side walls + far wall (blue) and a translucent top cap marking the
# trigger altitude (orange), matching the TikZ half-space shading.
# ---------------------------------------------------------------------------
const YMAX  = 0.120
const ZTRIG = p.trigger_altitude
hwid(y)     = aw + slope * y

funnel_bot = [Point3f( hwid(0.0),  0.0,  FLOORZ), Point3f(-hwid(0.0),  0.0,  FLOORZ),
              Point3f(-hwid(YMAX), YMAX, FLOORZ), Point3f( hwid(YMAX), YMAX, FLOORZ)]
funnel_top = [Point3f(c[1], c[2], ZTRIG) for c in funnel_bot]

# --- scene bounds ----------------------------------------------------------
# leave headroom above the top CoM for the (tilted) booster nose so it isn't clipped
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

# palette. structural elements (tower + arms) use the same light-slate that the
# hopper / quadrotor GLMakie plots use for their meshes; the keep-in corridor and
# trigger cap use the blue / gold from the quadrotor plot's GATE_COLORS
# (GATE_COLORS[1] = "#2C6FB0", GATE_COLORS[2] = "#E6A817").
const C_TOWER  = RGBf(0.68, 0.73, 0.80)      # light slate (shared scene colour)
const C_ARM    = RGBf(0.55, 0.60, 0.67)      # slightly darker slate, same family
const C_FUNNEL = RGBf(0x2C / 255, 0x6F / 255, 0xB0 / 255)  # quadrotor GATE_COLORS[1] (blue)
const C_TRIG   = RGBf(0xE6 / 255, 0xA8 / 255, 0x17 / 255)  # quadrotor GATE_COLORS[2] (gold)

fig = let
    fig = Figure(size = (760, 900), figure_padding = 0)
    ax  = Axis3(fig[1, 1], aspect = :data, perspectiveness = 0.8, limits = lims)
    hidedecorations!(ax); hidespines!(ax)

    set_ambient_light!(ax.scene, RGBf(0.65, 0.65, 0.65))

    # ground grid
    xy_grid!(ax; xlim = (lims[1], lims[2]), ylim = (lims[3], lims[4]), z = FLOORZ,
             step = 0.02, color = (:black, 0.15), lw = 1)

    # tower body + two splayed chopstick arms
    mesh!(ax, box_mesh(-TXW, TXW, -TDEP, 0.0, FLOORZ, TOWER_H); color = C_TOWER, LIT...)
    for s in (-1.0, 1.0)
        mesh!(ax, prism_mesh(arm_base(s), ARM_THK); color = C_ARM, LIT...)
    end

    # keep-in corridor: translucent slanted walls (+x, −x) and far wall, drawn as
    # the TikZ half-space shading. the near (tower-face) wall is left open.
    for (i, j) in ((1, 4), (2, 3), (4, 3))     # +x wall, −x wall, far wall
        mesh!(ax, quad_mesh(funnel_bot[i], funnel_bot[j], funnel_top[j], funnel_top[i]);
              color = RGBAf(C_FUNNEL.r, C_FUNNEL.g, C_FUNNEL.b, 0.14),
              transparency = true, shading = false)
    end
    # vertical corridor edges
    for i in 1:4
        lines!(ax, [funnel_bot[i], funnel_top[i]]; color = (C_FUNNEL, 0.9),
               linewidth = 1.4, linestyle = :dash)
    end
    lines!(ax, vcat(funnel_bot, [funnel_bot[1]]); color = (C_FUNNEL, 0.9),
           linewidth = 1.2, linestyle = :dash)

    # trigger-altitude top cap (orange, translucent)
    mesh!(ax, quad_mesh(funnel_top...); color = RGBAf(C_TRIG.r, C_TRIG.g, C_TRIG.b, 0.35),
          transparency = true, shading = false)
    lines!(ax, vcat(funnel_top, [funnel_top[1]]); color = C_TRIG, linewidth = 2.0)

    # CoM descent trajectory, shaded by a time gradient. custom cool→warm ramp
    # routed through purple/magenta (not green) so it keeps a blue first glyph and
    # a red catch but never passes through the turquoise a blue→green rainbow does.
    rocket_grad = cgrad([RGBf(0x2C / 255, 0x6F / 255, 0xB0 / 255),   # blue
                         RGBf(0x7A / 255, 0x4F / 255, 0xB5 / 255),   # violet
                         RGBf(0xC1 / 255, 0x3C / 255, 0x9A / 255),   # magenta
                         RGBf(0xE6 / 255, 0x7E / 255, 0x22 / 255),   # orange
                         RGBf(0xCB / 255, 0x41 / 255, 0x54 / 255)])  # red (catch)
    tcolors = [rocket_grad[t] for t in range(0, 1; length = N)]
    lines!(ax, Point3f.(P); color = :black, joinstyle = :round, linecap = :round,
           linewidth = 1.2, alpha = 0.8)

    # booster glyph placement: one at the start, a couple spread by arc length
    # over the coasting descent above the trigger, one at the trigger-altitude
    # crossing (where the booster straightens out into the constrained region),
    # and one at the final catch.
    seglen = [norm(P[k+1] .- P[k]) for k in 1:N-1]
    arc    = vcat(0.0, cumsum(seglen))                       # cumulative arc length
    k_trig = something(findfirst(k -> P[k][3] <= ZTRIG, 1:N), N)  # trigger crossing
    n_mid  = 3                                               # intermediates above trigger
    mid_ks = [argmin(abs.(arc .- f * arc[k_trig])) for f in range(0.24, 0.72; length = n_mid)]
    rocket_ks = unique!([1; mid_ks; k_trig; N])
    for i in rocket_ks
        plot_rocket!(ax, P[i], qs[i]; color = tcolors[i], δr = X[i, 13], δp = X[i, 14])
    end

    # press "p" to print the current camera angles (handy for tuning the save view)
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press && event.key == Keyboard.p
            @show ax.azimuth[]
            @show ax.elevation[]
        end
    end

    # high, front-left, looking-down view matching the TikZ "high_left_r2" camera
    ax.azimuth[]   = 2.507
    ax.elevation[] = 0.452
    fig
end

const OUTPATH = joinpath(@__DIR__, "png_renders", "rocket_catch_3d_glmakie.png")
mkpath(dirname(OUTPATH))
save(OUTPATH, fig; px_per_unit = 6)

# --- trim the surrounding whitespace ---------------------------------------
let img = load(OUTPATH)
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
