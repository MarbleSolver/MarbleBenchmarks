using JuMP, NLPModelsJuMP
using LinearAlgebra
using MarbleBenchmarks

##

problem = get_problem(:rocket_catch; N=40, T=4.5);

##
using GLMakie

function rocket_solution_components(problem, z)
    p = problem.params
    x = z[problem.inds[:x]]
    u = z[problem.inds[:u]]
    pos = x[:, 1:3] .+ p.x̄[1:3]'
    ϕ = x[:, 4:6]

    q = [MarbleBenchmarks.L(p.x̄[4:7]) * MarbleBenchmarks.rptoq(ϕ[k, :]) for k in 1:p.N]
    R = [MarbleBenchmarks.qtoQ(q[k]) for k in 1:p.N]

    thrust     = p.ū[1] .+ u[:, 1]
    gimbal     = u[:, 2:3]
    thrust_dir = [R[k] * MarbleBenchmarks.rocket_thrust_axis_body(gimbal[k, 1], gimbal[k, 2]) for k in 1:p.N]
    nozzle_dir = [-thrust_dir[k] for k in 1:p.N]

    return (; x, u, pos, ϕ, q, R, thrust, gimbal, thrust_dir, nozzle_dir)
end

function plane_surface(a, b, xlims, ylims, zlims)
    if abs(a[3]) > max(abs(a[1]), abs(a[2]))
        xs = [xlims[1] xlims[2]; xlims[1] xlims[2]]
        ys = [ylims[1] ylims[1]; ylims[2] ylims[2]]
        zs = (b .- a[1] .* xs .- a[2] .* ys) ./ a[3]
    elseif abs(a[2]) >= abs(a[1])
        xs = [xlims[1] xlims[2]; xlims[1] xlims[2]]
        zs = [zlims[1] zlims[1]; zlims[2] zlims[2]]
        ys = (b .- a[1] .* xs .- a[3] .* zs) ./ a[2]
    else
        ys = [ylims[1] ylims[2]; ylims[1] ylims[2]]
        zs = [zlims[1] zlims[1]; zlims[2] zlims[2]]
        xs = (b .- a[2] .* ys .- a[3] .* zs) ./ a[1]
    end
    return xs, ys, zs
end

function horizontal_surface(z, xlims, ylims)
    xs = [xlims[1] xlims[2]; xlims[1] xlims[2]]
    ys = [ylims[1] ylims[1]; ylims[2] ylims[2]]
    zs = fill(z, 2, 2)
    return xs, ys, zs
end

function add_plane_border!(ax, xs, ys, zs; color, linewidth=2)
    corners = Point3f[
        Point3f(xs[1, 1], ys[1, 1], zs[1, 1]),
        Point3f(xs[1, 2], ys[1, 2], zs[1, 2]),
        Point3f(xs[2, 2], ys[2, 2], zs[2, 2]),
        Point3f(xs[2, 1], ys[2, 1], zs[2, 1]),
        Point3f(xs[1, 1], ys[1, 1], zs[1, 1]),
    ]
    lines!(ax, corners; color, linewidth)
end

function plot_quad_plane!(ax, xs, ys, zs; color, alpha, label="")
    verts = Point3f[
        Point3f(xs[1, 1], ys[1, 1], zs[1, 1]),
        Point3f(xs[1, 2], ys[1, 2], zs[1, 2]),
        Point3f(xs[2, 2], ys[2, 2], zs[2, 2]),
        Point3f(xs[2, 1], ys[2, 1], zs[2, 1]),
    ]
    mesh!(ax, verts, [1 2 3; 1 3 4]; color=(color, alpha), transparency=true, label)
end

function plot_plane!(ax, a, b, xlims, ylims, zlims; color, alpha=0.18, label="")
    xs, ys, zs = plane_surface(a, b, xlims, ylims, zlims)
    plot_quad_plane!(ax, xs, ys, zs; color, alpha, label)
    add_plane_border!(ax, xs, ys, zs; color=(color, 0.65), linewidth=1.5)
    return xs, ys, zs
end

function plot_horizontal_plane!(ax, z, xlims, ylims; color, alpha=0.14, label="")
    xs, ys, zs = horizontal_surface(z, xlims, ylims)
    plot_quad_plane!(ax, xs, ys, zs; color, alpha, label)
    add_plane_border!(ax, xs, ys, zs; color=(color, 0.65), linewidth=1.5)
end

function draw_catch_hardware!(ax, p, ylims)
    z0 = 0.0
    z1 = max(p.trigger_altitude, p.catch_altitude) + 0.6
    tower_x = 0.0
    tower_y = p.tower_y
    tower_color = (:gray25, 0.95)

    for xoff in (-0.25, 0.25)
        lines!(ax, [Point3f(tower_x + xoff, tower_y, z0), Point3f(tower_x + xoff, tower_y, z1)];
               color=tower_color, linewidth=5)
    end
    lines!(ax,
        [Point3f(tower_x - 0.45, tower_y, p.catch_altitude), Point3f(tower_x + 0.45, tower_y, p.catch_altitude)];
        color=tower_color, linewidth=5)

    yarm = range(max(tower_y, ylims[1]), ylims[2]; length=40)
    left  = [Point3f(p.catch_b[2] - p.catch_A[2, 2] * y, y, p.catch_altitude) for y in yarm]
    right = [Point3f(-(p.catch_b[3] - p.catch_A[3, 2] * y), y, p.catch_altitude) for y in yarm]
    lines!(ax, left;  color=:dodgerblue3, linewidth=5, label="catch arm planes")
    lines!(ax, right; color=:dodgerblue3, linewidth=5)
end

function draw_rocket_poses!(ax, comp, p; stride=1)
    body_color = (:black, 0.42)
    nose_color = (:black, 0.45)

    for k in 1:stride:p.N
        r = comp.R[k]
        c = comp.pos[k, :]
        nose = c + r * [0.0, 0.0, p.l]
        tail = c - r * [0.0, 0.0, p.l]
        lines!(ax, [Point3f(tail), Point3f(nose)]; color=body_color, linewidth=2.5)
        scatter!(ax, [Point3f(nose)]; color=nose_color, markersize=6)
    end
end

function plot_rocket_catch_solution(problem, z;
        title="Rocket Catch",
        thrust_scale=0.55,
        pose_stride=1,
        thrust_stride=1,
        show_nozzle_clearance=true)
    p = problem.params
    comp = rocket_solution_components(problem, z)

    xmin = min(minimum(comp.pos[:, 1]), -2.0) - 0.75
    xmax = max(maximum(comp.pos[:, 1]),  2.0) + 0.75
    ymin = min(minimum(comp.pos[:, 2]), p.tower_y - 0.75) - 0.75
    ymax = max(maximum(comp.pos[:, 2]), p.tower_y + 3.0) + 0.75
    zmin = min(0.0, minimum(comp.pos[:, 3]) - 0.5)
    zmax = max(maximum(comp.pos[:, 3]), p.trigger_altitude) + 0.9
    xlims, ylims, zlims = (xmin, xmax), (ymin, ymax), (zmin, zmax)

    fig = Figure(size=(1150, 820))
    ax = Axis3(fig[1, 1];
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        title=title,
        aspect=:data,
        limits=(xlims, ylims, zlims),
        perspectiveness=0.65)

    plane_colors = [:dodgerblue3, :seagreen3, :seagreen3]
    plane_labels = ["tower front halfspace", "left catch-arm halfspace", "right catch-arm halfspace"]
    for j in axes(p.catch_A, 1)
        plot_plane!(ax, p.catch_A[j, :], p.catch_b[j], xlims, ylims, zlims;
            color=plane_colors[j], alpha=j == 1 ? 0.12 : 0.18, label=plane_labels[j])
    end
    plot_horizontal_plane!(ax, p.trigger_altitude, xlims, ylims;
        color=:purple, alpha=0.12, label="trigger altitude")
    plot_horizontal_plane!(ax, p.catch_altitude, xlims, ylims;
        color=:goldenrod, alpha=0.10, label="catch altitude")
    draw_catch_hardware!(ax, p, ylims)

    traj = [Point3f(comp.pos[k, 1], comp.pos[k, 2], comp.pos[k, 3]) for k in 1:p.N]
    active = comp.pos[:, 3] .<= p.trigger_altitude
    lines!(ax, traj; color=:black, linewidth=3, label="trajectory")
    !isempty(traj[.!active]) && scatter!(ax, traj[.!active]; color=:gray35, markersize=8,  label="above trigger")
    !isempty(traj[active])   && scatter!(ax, traj[active];   color=:purple, markersize=14, marker=:diamond, label="trigger active")
    scatter!(ax, [traj[1]];   color=:black, markersize=16, marker=:circle, label="start")
    scatter!(ax, [traj[end]]; color=:gold,  markersize=20, marker=:star5,  label="terminal")

    draw_rocket_poses!(ax, comp, p; stride=pose_stride)

    thrust_origins = Point3f[]
    thrust_vectors = Vec3f[]
    nozzle_origins = Point3f[]
    nozzle_vectors = Vec3f[]
    max_thrust = max(maximum(abs.(comp.thrust)), eps())
    for k in 1:thrust_stride:p.N
        tail = comp.pos[k, :] - comp.R[k] * [0.0, 0.0, p.l]
        push!(thrust_origins, Point3f(tail))
        push!(thrust_vectors, Vec3f(thrust_scale * comp.thrust[k] / max_thrust * comp.thrust_dir[k]))
        if show_nozzle_clearance
            push!(nozzle_origins, Point3f(tail))
            push!(nozzle_vectors, Vec3f(0.35 * comp.nozzle_dir[k]))
        end
    end
    arrows3d!(ax, thrust_origins, thrust_vectors; color=(:orange, 0.85), lengthscale=1.0, label="thrust")
    if show_nozzle_clearance
        arrows3d!(ax, nozzle_origins, nozzle_vectors; color=(:red, 0.35), lengthscale=1.0, label="engine nozzle")
    end

    text!(ax, Point3f(xlims[1], ylims[2], p.trigger_altitude);
        text="trigger altitude", color=:purple,    fontsize=15, align=(:left, :bottom))
    text!(ax, Point3f(xlims[1], ylims[2], p.catch_altitude);
        text="catch height",    color=:goldenrod4, fontsize=15, align=(:left, :bottom))

    axislegend(ax; position=:rb)
    display(fig)
    return fig
end

## Gurobi

using Gurobi

gurobi_model = reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)
set_optimizer(gurobi_model, Gurobi.Optimizer)
optimize!(gurobi_model)

status = termination_status(gurobi_model)
primal  = primal_status(gurobi_model)

println("Gurobi status: $(status)")
println("Gurobi primal status: $(primal)")
println("Gurobi solve time: $(solve_time(gurobi_model))")

if primal == FEASIBLE_POINT
    z_gurobi = value.(all_variables(gurobi_model))[1:num_variables(problem.model)]
else
    error("Gurobi did not find a feasible solution.")
end

##

plot_rocket_catch_solution(problem, z_gurobi; title="Rocket Catch (Gurobi)")

## Marble

using Marble

nlp = MathOptNLPModel(problem.model)
marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
Marble.max_iters!(opts, 4000)
result = Marble.solve(marble_data; opts=opts)

@show Marble.converged(result)
@show Marble.solve_time_s(result)

z_marble = Marble.z(result)

##

plot_rocket_catch_solution(problem, z_marble; title="Rocket Catch (Marble)")
