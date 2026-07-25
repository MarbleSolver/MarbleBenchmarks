using GLMakie
using MarbleBenchmarks
using Serialization
using LinearAlgebra
using GLMakie.FileIO: load
using GLMakie: Point3f, Vec3f, RGBAf, RGBf

GLMakie.activate!()
GLMakie.closeall() # close any open screen

const SOLDIR = joinpath(@__DIR__, "..", "benchmarks", "robotics", "results", "solutions")
latest_marble() = joinpath(SOLDIR,
    sort(filter(f -> startswith(f, "marble") && endswith(f, ".jls"), readdir(SOLDIR)))[end])

solfile = length(ARGS) >= 1 ? ARGS[1] : latest_marble()

# The benchmark solves this same `get_problem(:lane_passing_2d)`, so the params
# below match the saved solution. That matters beyond the geometry: cruiser speed
# feeds cruiser_x(), so a mismatch misplaces the cruiser and breaks the
# "alongside" detection.
problem = get_problem(:lane_passing_2d)
inds, p = problem.inds, problem.params

z = deserialize(solfile)["lane_passing_2d"]
X = z[inds[:x]]      # N × 4 state [px, py, θ, v]

(; N, Δt, cruiser_mps, lane_width, w_cruiser, h_cruiser, w_passer, h_passer) = p

px, py, θ = X[:, 1], X[:, 2], X[:, 3]
Nk = size(X, 1)
lw = lane_width
y_edge = 3 / 2 * lw                              # outer road boundary [m]

cruiser_x(i) = cruiser_mps * Δt * (i - 1)        # cruiser longitudinal position

# Use |py| so detection is side-agnostic (the passer may overtake on either side).
apy         = abs.(py)
knot_leave  = findfirst(>=(lw / 2), apy)          # passer crosses out of centre lane
knot_along  = argmin(abs.(px .- cruiser_x.(1:Nk)))   # side-by-side with cruiser
ipk         = argmax(apy)                          # apex of the pass
knot_rejoin = ipk - 1 + findfirst(<(lw / 2), apy[ipk:end])  # crosses back into centre lane
# first knot (start, behind cruiser) → leave lane → alongside → rejoin → last knot (ahead)
event_knots = (1, knot_leave, knot_along, knot_rejoin, Nk)

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

# shaded material with a restrained specular (matches the other GLMakie scripts).
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

# place a car mesh centred at (cx, cy) on the road, yawed by heading `ang`. `yaw0`
# absorbs a mesh's forward-axis convention: car-1 and car-2 are authored nose
# toward opposite ends of their local z, so the cruiser needs a 180° base yaw
# relative to the passer for both to drive down +x.
function place_car!(ax, meshobj, s, ymin; cx, cy, ang, color, yaw0 = 0.0)
    pl = mesh!(ax, meshobj; color = color, LIT...)
    GLMakie.scale!(pl, (s, s, s))
    GLMakie.rotate!(pl, makie_quat(Q_to_quat(Rz(ang + yaw0) * R_BASE)))
    GLMakie.translate!(pl, cx, cy, -s * ymin)      # local y-min → world z = 0
    return pl
end

# flat filled rectangle on the road, centred at (cx, cy) and yawed by `ang`
# (`len` along heading, `wid` across) — mirrors the rectangular bounding box the
# collision model actually uses, sitting just under an event car as a highlight.
function footprint!(ax, cx, cy, ang; len, wid, z = 0.03, color)
    hl, hw = len / 2, wid / 2
    c, s = cos(ang), sin(ang)
    corners = ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))
    verts = [Point3f(cx + c*a - s*b, cy + s*a + c*b, z) for (a, b) in corners]
    faces = [GLMakie.GeometryBasics.TriangleFace(1, 2, 3),
             GLMakie.GeometryBasics.TriangleFace(1, 3, 4)]
    mesh!(ax, verts, faces; color = color, shading = false, transparency = true)
end

# world-aligned footprint (length along local z, width along local x) of a scaled mesh
function mesh_footprint(meshobj, s)
    pts = GLMakie.GeometryBasics.coordinates(meshobj)
    xloc = [pt[1] for pt in pts]; zloc = [pt[3] for pt in pts]
    return (maximum(zloc) - minimum(zloc)) * s, (maximum(xloc) - minimum(xloc)) * s
end

# highlight box = the passer's footprint, enlarged slightly so it peeks out
passer_fp_len, passer_fp_wid = mesh_footprint(passer_mesh, passer_s) .* 1.12

# rainbow time gradient shared by the path line and the event cars
tcolors = cgrad(:diverging_rainbow_bgymr_45_85_c67_n256, Nk; categorical = true).colors

fig = let
    # span both cars across every drawn knot (passer starts behind, ends ahead)
    x_lo = min(minimum(px[collect(event_knots)]), cruiser_x(1)) - 8.0
    x_hi = max(maximum(px[collect(event_knots)]), cruiser_x(Nk)) + 8.0
    lims = (x_lo, x_hi, -y_edge - 1.0, y_edge + 1.0, 0.0, 8.0)

    fig = Figure(size = (1600, 700), backgroundcolor = :white)
    ax  = Axis3(fig[1, 1], aspect = :data, perspectiveness = 0.35, limits = lims,
                azimuth = -0.1328515625, elevation = 0.1064843749999996)
    hidedecorations!(ax); hidespines!(ax)
    set_ambient_light!(ax.scene, RGBf(0.62, 0.62, 0.64))

    # pure white and flat (unshaded). Note the trim below keys white → transparent,
    # so in the saved PNG the road becomes transparent and reads as totally white on
    # any white/light background; the grey barriers still fix the crop bounding box.
    road = Point3f[(x_lo, -y_edge, 0), (x_hi, -y_edge, 0), (x_hi, y_edge, 0), (x_lo, y_edge, 0)]
    mesh!(ax, road, [1 2 3; 1 3 4]; color = RGBf(1.0, 1.0, 1.0), shading = false)

    # lane markings, lifted slightly to avoid z-fighting with the road
    zm = 0.02
    for yb in (-y_edge, y_edge)                    # thin light-grey solid side edges
        lines!(ax, [x_lo, x_hi], [yb, yb], [zm, zm]; color = RGBf(0.72, 0.72, 0.72), linewidth = 2.5)
    end
    for yb in (-lw/2, lw/2)                         # light-grey dashed lane dividers
        lines!(ax, [x_lo, x_hi], [yb, yb], [zm, zm]; color = RGBf(0.72, 0.72, 0.72),
               linewidth = 2.5, linestyle = :dash)
    end

    lines!(ax, px, py, fill(0.05, Nk); color = 1:Nk,
           colormap = :diverging_rainbow_bgymr_45_85_c67_n256,
           linewidth = 4, joinstyle = :round)

    for k in event_knots
        ci = tcolors[k]
        # cruiser (grey) at this instant, holding the centre lane (skip the final
        # knot so the passer's last car isn't trailed by a redundant cruiser)
        if k != Nk
            place_car!(ax, cruiser_mesh, cruiser_s, cruiser_ymin;
                       cx = cruiser_x(k), cy = 0.0, ang = 0.0, yaw0 = 0.0,
                       color = RGBf(0.68, 0.73, 0.80))   # slate-blue (hopper staircase / inactive drone)
        end
        # passer, coloured by its time along the manoeuvre; the highlight box under
        # it matches the rectangular bounding box the collision model uses
        footprint!(ax, px[k], py[k], θ[k]; len = passer_fp_len, wid = passer_fp_wid,
                   color = RGBAf(ci.r, ci.g, ci.b, 0.35))
        place_car!(ax, passer_mesh, passer_s, passer_ymin;
                   cx = px[k], cy = py[k], ang = θ[k], yaw0 = Float64(π), color = ci)
    end

    # press "p" to print the current camera angles (handy for tuning the view)
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press && event.key == Keyboard.p
            @show ax.azimuth[]; @show ax.elevation[]
        end
    end

    ax.azimuth[] = -0.08640624999999999
    ax.elevation[] = 0.16925781249999977

    fig
end

# Interactive window (drag to orbit, scroll to zoom). Uncomment to inspect.
screen = display(fig)
isinteractive() || wait(screen)

const OUTPATH = joinpath(@__DIR__, "png_renders", "lane_passing_2d_glmakie.png")
mkpath(dirname(OUTPATH))
save(OUTPATH, fig; px_per_unit = 6)

#= transparency + crop re-save disabled for now
# --- trim whitespace and make the background transparent --------------------
# Crop the rendered PNG to the bounding box of the non-white content (with a
# small margin) and key the near-white background out to alpha 0, softening
# anti-aliased edges with a narrow alpha ramp just below the white cutoff.
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
