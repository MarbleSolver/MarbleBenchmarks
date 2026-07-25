using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie: Point3f, Vec3f, RGBAf, RGBf
import GLMakie.Makie as MK      # scene lights + camera control

GLMakie.activate!()
GLMakie.closeall() # close any open screen

const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

problem = get_problem(:lane_passing_2d)
inds, p = problem.inds, problem.params

z = deserialize(solfile)["lane_passing_2d"]
X = z[inds[:x]]      # N × 4 state [px, py, θ, v]

(; N, Δt, cruiser_mps, lane_width, w_cruiser, h_cruiser, w_passer, h_passer) = p

px, py, θ = X[:, 1], X[:, 2], X[:, 3]
Nk = size(X, 1)
lw = lane_width
y_edge = 3 / 2 * lw                              # outer road boundary [m]

cruiser_x(s) = cruiser_mps * Δt * (s - 1)        # cruiser longitudinal position (continuous)

# linear interpolation of the passer state at a continuous knot parameter s ∈ [1, Nk]
function passer_at(s::Real)
    s = clamp(float(s), 1.0, float(Nk))
    k0 = floor(Int, s); k1 = min(k0 + 1, Nk); a = s - k0
    return ((1 - a) * px[k0] + a * px[k1],
            (1 - a) * py[k0] + a * py[k1],
            (1 - a) * θ[k0]  + a * θ[k1])
end

# rainbow time gradient shared by the path line and the passer tint
const TGRAD = cgrad(:diverging_rainbow_bgymr_45_85_c67_n256)
tcolor(frac) = TGRAD[clamp(float(frac), 0.0, 1.0)]

function Q_to_quat(R)
    tr = R[1,1] + R[2,2] + R[3,3]
    if tr > 0
        S = sqrt(tr + 1.0) * 2
        w = 0.25 * S; x = (R[3,2]-R[2,3])/S; y = (R[1,3]-R[3,1])/S; z = (R[2,1]-R[1,2])/S
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        S = sqrt(1.0 + R[1,1] - R[2,2] - R[3,3]) * 2
        w = (R[3,2]-R[2,3])/S; x = 0.25*S; y = (R[1,2]+R[2,1])/S; z = (R[1,3]+R[3,1])/S
    elseif R[2,2] > R[3,3]
        S = sqrt(1.0 + R[2,2] - R[1,1] - R[3,3]) * 2
        w = (R[1,3]-R[3,1])/S; x = (R[1,2]+R[2,1])/S; y = 0.25*S; z = (R[2,3]+R[3,2])/S
    else
        S = sqrt(1.0 + R[3,3] - R[1,1] - R[2,2]) * 2
        w = (R[2,1]-R[1,2])/S; x = (R[1,3]+R[3,1])/S; y = (R[2,3]+R[3,2])/S; z = 0.25*S
    end
    [w, x, y, z]
end
makie_quat(q) = GLMakie.Quaternion(q[2], q[3], q[4], q[1])
Rz(a) = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1]

# shaded material with a restrained specular (matches the still).
const LIT = (; shading = true, specular = 0.15f0, shininess = 20.0f0)

# both assets are authored Y-up / Z-forward; R_BASE maps local→world so the car's
# forward axis (local z) points down the road (world x) and its up axis (local y)
# points to world z.
const R_BASE = [0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
cruiser_mesh = load(joinpath(@__DIR__, "assets", "car-1.obj"))   # Cinema 4D sedan
passer_mesh  = load(joinpath(@__DIR__, "assets", "car-2.obj"))   # Blender sedan

# vertical scale so `mesh` fills `len` metres along its forward (local z) axis,
# plus the local y-min needed to drop the wheels onto the road (world z = 0).
function mesh_fit(meshobj, len)
    pts = GLMakie.GeometryBasics.coordinates(meshobj)
    zloc = [pt[3] for pt in pts]; yloc = [pt[2] for pt in pts]
    s = len / (maximum(zloc) - minimum(zloc))
    return s, minimum(yloc)
end
cruiser_s, cruiser_ymin = mesh_fit(cruiser_mesh, h_cruiser * 0.62)  # visual length ≈ 5.6 m
passer_s,  passer_ymin  = mesh_fit(passer_mesh,  h_passer)          # visual length = 5.0 m

# orient/place a (single, reused) car mesh centred at (cx, cy) on the road, yawed by
# `ang`. `yaw0` absorbs the mesh's forward-axis convention (car-1 and car-2 point
# opposite ways along local z, so the cruiser needs a 180° base yaw vs the passer).
function set_car!(pl, s, ymin; cx, cy, ang, yaw0 = 0.0)
    GLMakie.rotate!(pl, makie_quat(Q_to_quat(Rz(ang + yaw0) * R_BASE)))
    GLMakie.translate!(pl, cx, cy, -s * ymin)      # local y-min → world z = 0
end

# These are the actual polytopes the growth-distance MPCC keeps apart: axis-aligned
# boxes in the world frame, centred on each car. Half-extents come straight from the
# solved model (b = [±length; ±width]/2). The passer's box uses the *effective*
# dimensions (enlarged for its ≤10° heading) so an axis-aligned box still contains
# the turned car — which is why it never rotates with the heading.
cbx, cby = p.b_cruiser[1], p.b_cruiser[3]   # cruiser half-length (x), half-width (y)
pbx, pby = p.b_passer[1],  p.b_passer[3]    # passer  half-length (x), half-width (y)

# a box height, taken from each car's visual height so the 3-D box encloses the mesh
function mesh_height(meshobj, s)
    yl = [pt[2] for pt in GLMakie.GeometryBasics.coordinates(meshobj)]
    (maximum(yl) - minimum(yl)) * s
end
cruiser_H = mesh_height(cruiser_mesh, cruiser_s)
passer_H  = mesh_height(passer_mesh,  passer_s)

# 24 endpoints (12 edges) of an axis-aligned box centred at (cx, cy), for linesegments!
function box_edges(cx, cy, hx, hy, z0, z1)
    b1 = Point3f(cx-hx, cy-hy, z0); b2 = Point3f(cx+hx, cy-hy, z0)
    b3 = Point3f(cx+hx, cy+hy, z0); b4 = Point3f(cx-hx, cy+hy, z0)
    t1 = Point3f(cx-hx, cy-hy, z1); t2 = Point3f(cx+hx, cy-hy, z1)
    t3 = Point3f(cx+hx, cy+hy, z1); t4 = Point3f(cx-hx, cy+hy, z1)
    Point3f[b1,b2, b2,b3, b3,b4, b4,b1,   # bottom rectangle
            t1,t2, t2,t3, t3,t4, t4,t1,   # top rectangle
            b1,t1, b2,t2, b3,t3, b4,t4]   # verticals
end

const x_lo = min(px[1], cruiser_x(1)) - 50.0
const x_hi = max(px[Nk], cruiser_x(Nk)) + 160.0

# A real 3-D camera (cam3d! on a bare Scene) follows the manoeuvre from behind and
# above at a low angle (matching the still's chase view). It looks a little ahead of
# the midpoint of the two cars, so both the passer and the cruiser stay framed as
# the passer moves from behind the cruiser to ahead of it. Because the camera is a
# real camera (not an Axis3 data box), nothing is clipped and the view fills the
# frame — no wide margin of whitespace.
const CAM_AZ     = Float64(π) + 0.13   # mostly behind, slight 3/4 to one side
const CAM_EL     = 0.20                # low chase angle (a few degrees above the road)
const CAM_DIST   = 29.0                # eye distance from the look point [m] (back enough that
                                       # the nearer car stays fully framed at the pass extremes)
const LOOK_AHEAD = 5.0                 # look this far ahead of the two-car midpoint [m]
const LOOK_Z     = 1.1                 # look at roughly car height [m]
const CAM_UP     = Vec3f(0, 0, 1)
const CAM_DIR    = Vec3f(cos(CAM_EL)*cos(CAM_AZ), cos(CAM_EL)*sin(CAM_AZ), sin(CAM_EL))
cam_lookat(s) = Point3f((passer_at(s)[1] + cruiser_x(s)) / 2 + LOOK_AHEAD, 0.0, LOOK_Z)
cam_eye(s)    = cam_lookat(s) + CAM_DIST * CAM_DIR

const FPS     = 60          # video frame rate (high, for smooth motion)
const SLOWMO  = 1.5         # video runs this much longer than real time (<1 speeds up)
const START_S = 0.6         # short hold on the initial pose
const END_S   = 1.2         # hold on the final pose

T_total  = (Nk - 1) * Δt                     # manoeuvre duration [s]
n_motion = max(2, round(Int, T_total * SLOWMO * FPS))
n_start  = round(Int, START_S * FPS)
n_end    = round(Int, END_S * FPS)
n_frames = n_start + n_motion + n_end

# knot parameter s and time fraction for each frame
function sfor(i)
    i <= n_start            && return 1.0
    i > n_start + n_motion  && return float(Nk)
    return 1 + (i - n_start - 1) / (n_motion - 1) * (Nk - 1)
end
fracfor(i) = (sfor(i) - 1) / (Nk - 1)
motion_idx(i) = clamp(i - n_start, 1, n_motion)   # index into the precomputed path

# precompute the passer's rear-axle path over the motion frames (for the growing trail)
path_pts = [(let (x, y, _) = passer_at(sfor(n_start + j)); Point3f(x, y, 0.05) end) for j in 1:n_motion]

lights = [MK.AmbientLight(RGBf(0.60, 0.60, 0.62)),
          MK.DirectionalLight(RGBf(0.95, 0.95, 0.95), Vec3f(-0.3, -0.4, -1.0))]
scene = Scene(size = (1700, 620), backgroundcolor = :white, lights = lights)
cam3d!(scene; fov = 38f0, near = 0.1f0, far = 5000f0, clipping_mode = :static)

road = Point3f[(x_lo, -y_edge, 0), (x_hi, -y_edge, 0), (x_hi, y_edge, 0), (x_lo, y_edge, 0)]
mesh!(scene, road, [1 2 3; 1 3 4]; color = RGBf(1.0, 1.0, 1.0), shading = false)

# lane markings, lifted slightly to avoid z-fighting with the road
let zm = 0.02
    for yb in (-y_edge, y_edge)                    # thin light-grey solid side edges
        lines!(scene, [x_lo, x_hi], [yb, yb], [zm, zm]; color = RGBf(0.72, 0.72, 0.72), linewidth = 2.5)
    end
    for yb in (-lw/2, lw/2)                         # light-grey dashed lane dividers
        lines!(scene, [x_lo, x_hi], [yb, yb], [zm, zm]; color = RGBf(0.72, 0.72, 0.72),
               linewidth = 2.5, linestyle = :dash)
    end
end

trail_pts = Observable(Point3f[path_pts[1]])
lines!(scene, trail_pts; color = :black, joinstyle = :round, linecap = :round,
       linewidth = 1.2, alpha = 0.8)

# cruiser: grey slate, holds the centre lane
cruiser_pl = mesh!(scene, cruiser_mesh; color = RGBf(0.68, 0.73, 0.80), LIT...)
GLMakie.scale!(cruiser_pl, (cruiser_s, cruiser_s, cruiser_s))
set_car!(cruiser_pl, cruiser_s, cruiser_ymin; cx = cruiser_x(1.0), cy = 0.0, ang = 0.0)

# passer: tinted by its time along the manoeuvre
passer_col = Observable{Any}(RGBf(tcolor(0.0)))
passer_pl  = mesh!(scene, passer_mesh; color = passer_col, LIT...)
GLMakie.scale!(passer_pl, (passer_s, passer_s, passer_s))
let (x, y, a) = passer_at(1.0)
    set_car!(passer_pl, passer_s, passer_ymin; cx = x, cy = y, ang = a, yaw0 = Float64(π))
end

cbox = Observable(box_edges(cruiser_x(1.0), 0.0, cbx, cby, 0.02, cruiser_H))
linesegments!(scene, cbox; color = RGBf(0.42, 0.47, 0.53), linewidth = 2.5)
pbox = Observable(box_edges(passer_at(1.0)[1], passer_at(1.0)[2], pbx, pby, 0.02, passer_H))
linesegments!(scene, pbox; color = passer_col, linewidth = 2.5)

# initial camera (updated every frame in the record loop below)
update_cam!(scene, cam_eye(1.0), cam_lookat(1.0), CAM_UP)

const OUTPATH = joinpath(@__DIR__, "animations", "lane_passing_2d_glmakie.mp4")
mkpath(dirname(OUTPATH))

record(scene, OUTPATH, 1:n_frames; framerate = FPS) do i
    s = sfor(i)
    x, y, a = passer_at(s)

    # cars
    set_car!(cruiser_pl, cruiser_s, cruiser_ymin; cx = cruiser_x(s), cy = 0.0, ang = 0.0)
    set_car!(passer_pl, passer_s, passer_ymin; cx = x, cy = y, ang = a, yaw0 = Float64(π))
    passer_col[] = RGBf(tcolor(fracfor(i)))

    # collision boxes track their cars (axis-aligned in the world frame)
    cbox[] = box_edges(cruiser_x(s), 0.0, cbx, cby, 0.02, cruiser_H)
    pbox[] = box_edges(x, y, pbx, pby, 0.02, passer_H)

    # growing path
    trail_pts[] = path_pts[1:motion_idx(i)]

    # chase camera: follow the two-car midpoint from behind/above
    update_cam!(scene, cam_eye(s), cam_lookat(s), CAM_UP)
end

@info "saved animation" OUTPATH n_frames FPS
