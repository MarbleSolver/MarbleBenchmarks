using PGFPlotsX, LinearAlgebra

function body_frame(d)
    d = normalize(d)
    a = abs(d[3]) < 0.9 ? [0.0, 0.0, 1.0] : [1.0, 0.0, 0.0]
    u = normalize(cross(d, a)); v = cross(d, u)
    d, u, v
end

"""Cylinder from p0 to p1, radius r, as a surf-ready Table."""
function cylinder_table(p0, p1, r; nθ = 40, ns = 2, light = normalize([1.0, 1.0, 1.5]))
    d, u, v = body_frame(p1 - p0)
    L = norm(p1 - p0)
    X, Y, Z, M = Float64[], Float64[], Float64[], Float64[]
    for s in range(0, 1, length = ns), θ in range(0, 2π, length = nθ)
        n = cos(θ) * u + sin(θ) * v          # outward normal
        pt = p0 + s * L * d + r * n
        push!(X, pt[1]); push!(Y, pt[2]); push!(Z, pt[3])
        push!(M, max(dot(n, light), 0.0))    # Lambertian shading
    end
    Table({"mesh/rows" => nθ}, [:x => X, :y => Y, :z => Z, :meta => M])
end

"""Cone from base circle at p1 to apex."""
function cone_table(p1, apex, r; nθ = 40, light = normalize([1.0, 1.0, 1.5]))
    d, u, v = body_frame(apex - p1)
    X, Y, Z, M = Float64[], Float64[], Float64[], Float64[]
    for s in (0.0, 1.0), θ in range(0, 2π, length = nθ)
        n = cos(θ) * u + sin(θ) * v
        pt = s == 0 ? p1 + r * n : apex
        push!(X, pt[1]); push!(Y, pt[2]); push!(Z, pt[3])
        push!(M, max(dot(normalize(n + 0.5 * d), light), 0.0))
    end
    Table({"mesh/rows" => nθ}, [:x => X, :y => Y, :z => Z, :meta => M])
end

p0, p1 = [0.0, 0.0, 0.0], [0.3, 0.2, 1.2]
d = normalize(p1 - p0)
r = 0.06

style = @pgf {
    surf,
    shader = "interp",
    "z buffer" = "sort",          # per-patch painter sort — the important bit
    colormap = "{gr}{gray(0cm)=(0.25); gray(1cm)=(0.95)}",
    "point meta" = raw"\thisrow{meta}",
    "forget plot",
}

fig = @pgf Axis(
    {
        width = "8cm", height = "8cm",
        view = "{40}{18}",
        xlabel = raw"$x$", ylabel = raw"$y$", zlabel = raw"$z$",
        zmin = 0, "axis equal" = true,
        grid = "both",
    },
    Plot3(style, cylinder_table(p0, p1, r)),
    Plot3(style, cone_table(p1, p1 + 0.25 * d, r)),
)

pgfsave("rocket.pdf", fig)