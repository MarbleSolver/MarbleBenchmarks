using JuMP, NLPModelsJuMP
using LinearAlgebra
using MarbleBenchmarks

##

problem = get_problem(Symbol("3d_gates_fewer_comps"); N=30, T=10.0, ngates=4, gate_violation_weight=1.0);

##
using GLMakie

function gate_corners_3d(pos, R, dims)
    hw, hh = dims[1] / 2, dims[3] / 2
    corners_gate = [[-hw, 0.0, -hh], [hw, 0.0, -hh], [hw, 0.0, hh], [-hw, 0.0, hh]]
    return Point3f.(pos + R * c for c in corners_gate)
end

function three_d_gates_solution_components(problem, z)
    p = problem.params
    x = z[problem.inds[:x]]
    u = z[problem.inds[:u]]
    μ = z[problem.inds[:μ]]
    gate_violation = z[problem.inds[:gate_violation]]

    pos = x[:, 1:3]
    vel = x[:, 4:6]
    progress = cumsum(μ; dims=1)
    times = collect(range(0.0, p.T; length=p.N))
    interval_times = times[2:end]

    return (; x, u, μ, gate_violation, pos, vel, progress, times, interval_times)
end

function plot_three_d_gates_solution(problem, z; title="3D Gates")
    p = problem.params
    comp = three_d_gates_solution_components(problem, z)

    gate_colors = [Makie.wong_colors()[i] for i in 1:p.ngates]

    fig = Figure(size=(1250, 780))
    ax = Axis3(fig[1:3, 1];
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        title=title,
        aspect=:data)

    for j in 1:p.ngates
        corners = gate_corners_3d(p.gate_positions[j], p.gate_orientations[j], p.gate_dims)
        c = gate_colors[j]
        mesh!(ax, corners, [1 2 3; 1 3 4]; color=(c, 0.28), transparency=true)
        lines!(ax, [corners; corners[1:1]]; color=c, linewidth=3, label="Gate $j")
        text!(ax, Point3f(p.gate_positions[j]);
            text="Gate $j", fontsize=13, color=c, align=(:center, :bottom), offset=(0, 8))
    end

    traj = [Point3f(comp.pos[k, :]) for k in 1:p.N]
    lines!(ax, traj; color=:black, linewidth=2.5, label="trajectory")
    scatter!(ax, [traj[1]]; color=:black, markersize=13, marker=:circle, label="start")
    scatter!(ax, [traj[end]]; color=:black, markersize=16, marker=:star5, label="end")

    for j in 1:p.ngates
        active = comp.μ[:, j] .> 1.0e-4
        if any(active)
            scatter!(ax, traj[active]; color=gate_colors[j], markersize=13, marker=:diamond,
                label="progress gate $j")
        end
    end
    axislegend(ax; position=:lt)

    ax_progress = Axis(fig[1, 2];
        xlabel="time (s)", ylabel="progress",
        title="Cumulative Gate Progress")
    for j in 1:p.ngates
        lines!(ax_progress, comp.times, comp.progress[:, j];
            color=gate_colors[j], linewidth=2.5, label="Gate $j")
    end
    hlines!(ax_progress, [1.0]; color=(:gray35, 0.65), linestyle=:dash, linewidth=1.5)
    axislegend(ax_progress; position=:rb)

    ax_violation = Axis(fig[2, 2];
        xlabel="time (s)", ylabel="violation",
        title="Gate Violation")
    for j in 1:p.ngates
        lines!(ax_violation, comp.times, comp.gate_violation[:, j];
            color=gate_colors[j], linewidth=2.5, label="Gate $j")
    end
    hlines!(ax_violation, [0.0]; color=(:gray35, 0.65), linewidth=1.2)

    ax_control = Axis(fig[3, 2];
        xlabel="time (s)", ylabel="acceleration",
        title="Control")
    control_colors = [:firebrick3, :seagreen4, :purple4]
    labels = ["u_x", "u_y", "u_z"]
    for d in 1:3
        lines!(ax_control, comp.interval_times, comp.u[:, d];
            color=control_colors[d], linewidth=2.2, label=labels[d])
    end
    hlines!(ax_control, [0.0]; color=(:gray35, 0.65), linewidth=1.0)
    axislegend(ax_control; position=:rb)

    linkxaxes!(ax_progress, ax_violation, ax_control)
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

plot_three_d_gates_solution(problem, z_gurobi; title="3D Gates (Gurobi)")

## Marble

using Marble

solver = Marble.Solver()
Marble.setup!(solver, problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types;
              verbosity = 1, debug = false, max_iters=2000, max_inner_iters_per_subproblem=50,
              prox_scale=0.1, relaxation_min=1e-8, penalty_max=1e8, relaxation_scaling=0.1)
results = Marble.solve!(solver)
z_marble = Marble.z(results)

##

plot_three_d_gates_solution(problem, z_marble; title="3D Gates (Marble)")
