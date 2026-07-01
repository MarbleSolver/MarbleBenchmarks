using JuMP, NLPModelsJuMP
using LinearAlgebra
using Combinatorics
using MarbleBenchmarks
using GLMakie

##

problem = get_problem(:multi_robot_coordination; N=30, T=5.0, d_min=5.0);

##
function extract_multi_robot_solution(problem, z)
    p = problem.params
    x_flat = z[problem.inds[:x]]
    # x is declared as x[1:N, 1:n_robots, 1:4] — column-major reshape
    x_vals = reshape(x_flat, p.N, p.n_robots, 4)
    pos = x_vals[:, :, 1:2]  # N × n_robots × 2  (x, y)
    vel = x_vals[:, :, 3:4]  # N × n_robots × 2  (vx, vy)
    return (; pos, vel)
end

function l1_pair_distances(pos, pairs)
    N = size(pos, 1)
    dists = [
        [norm(pos[k, p1, :] - pos[k, p2, :], 1) for k in 1:N]
        for (p1, p2) in pairs
    ]
    return dists
end

function plot_multi_robot_solution(problem, z; title="Multi-Robot Coordination")
    p = problem.params
    comp = extract_multi_robot_solution(problem, z)

    times = range(0.0, p.T; length=p.N)
    robot_colors = [Makie.wong_colors()[i] for i in 1:p.n_robots]
    pair_colors  = [Makie.wong_colors()[i + p.n_robots] for i in 1:length(p.pairs)]

    fig = Figure(size=(1100, 600))

    # ── Left panel: 2D trajectories ──────────────────────────────────────────
    ax_traj = Axis(fig[1, 1];
        xlabel="x (m)", ylabel="y (m)",
        title=title,
        aspect=DataAspect())

    for i in 1:p.n_robots
        c = robot_colors[i]
        xs = comp.pos[:, i, 1]
        ys = comp.pos[:, i, 2]
        lines!(ax_traj, xs, ys; color=c, linewidth=2.5, label="Robot $i")
        scatter!(ax_traj, [xs[1]],   [ys[1]];   color=c, markersize=14, marker=:circle)
        scatter!(ax_traj, [xs[end]], [ys[end]];  color=c, markersize=14, marker=:star5)
        # label start/end
        text!(ax_traj, xs[1],   ys[1];   text="R$(i) start", fontsize=11,
              color=c, align=(:center, :top),   offset=(0, -6))
        text!(ax_traj, xs[end], ys[end]; text="R$(i) end",   fontsize=11,
              color=c, align=(:center, :bottom), offset=(0,  6))
    end

    # draw d_min separation circle around each robot's initial position as visual guide
    θ = range(0, 2π; length=100)
    for i in 1:p.n_robots
        cx, cy = comp.pos[1, i, 1], comp.pos[1, i, 2]
        lines!(ax_traj, cx .+ (p.d_min / 2) .* cos.(θ), cy .+ (p.d_min / 2) .* sin.(θ);
               color=(:gray60, 0.35), linewidth=1, linestyle=:dash)
    end

    axislegend(ax_traj; position=:lt)

    # ── Right panel: pairwise L1 distances over time ──────────────────────────
    ax_dist = Axis(fig[1, 2];
        xlabel="time (s)", ylabel="L1 distance (m)",
        title="Pairwise L1 Distances")

    dists = l1_pair_distances(comp.pos, p.pairs)
    for (idx, (p1, p2)) in enumerate(p.pairs)
        lines!(ax_dist, times, dists[idx];
               color=pair_colors[idx], linewidth=2.5, label="‖R$(p1) − R$(p2)‖₁")
    end

    # d_min threshold line
    hlines!(ax_dist, [p.d_min]; color=:red, linewidth=1.5, linestyle=:dash, label="d_min = $(p.d_min)")
    axislegend(ax_dist; position=:rt)

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

plot_multi_robot_solution(problem, z_gurobi; title="Multi-Robot Coordination (Gurobi)")

## Marble

using Marble

solver = Marble.Solver()
Marble.setup!(solver, problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types;
              verbosity = 1, debug = false, max_iters=1000, max_inner_iters_per_subproblem=1000,
              prox_scale=0.0, relaxation_min=1e-8, penalty_max=1e8, relaxation_scaling=0.1)
results = Marble.solve!(solver)
z_marble = Marble.z(results)
# nlp = MathOptNLPModel(problem.model)
# marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

# opts = Marble.SolverOptions()
# Marble.verbosity!(opts, 1)
# Marble.max_iters!(opts, 4000)
# result = Marble.solve(marble_data; opts=opts)

# @show Marble.converged(result)
# @show Marble.solve_time_s(result)

##

plot_multi_robot_solution(problem, z_marble; title="Multi-Robot Coordination (Marble)")
