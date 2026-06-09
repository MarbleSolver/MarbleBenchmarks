using JuMP, NLPModelsJuMP
using LinearAlgebra
using Combinatorics
using MarbleBenchmarks

##

problem = get_problem(:block_push_1d; N=40, T=4.0,
	                      initial_block_position=0.0, target_block_position=3.0,
	                      initial_pusher_gap=1.0);

##
using GLMakie

function block_pushing_solution_components(problem, z)
    p = problem.params
    x = z[problem.inds[:x]]
    u = z[problem.inds[:u]]
    contact_force = z[problem.inds[:contact_force]]

    block_pos = x[:, p.block_position_index]
    block_vel = x[:, p.block_velocity_index]
    pusher_pos = x[:, p.pusher_position_index]
    pusher_vel = x[:, p.pusher_velocity_index]

    block_left = block_pos .- p.block_half_width
    block_right = block_pos .+ p.block_half_width
    contact_gap = block_left .- pusher_pos

    times = collect(range(0.0, p.T; length=p.N))
    interval_times = times[2:end]

    return (; x, u, contact_force, block_pos, block_vel, pusher_pos, pusher_vel,
              block_left, block_right, contact_gap, times, interval_times)
end

function plot_block_pushing_solution(problem, z; title="1D Block Pushing")
    p = problem.params
    comp = block_pushing_solution_components(problem, z)

    fig = Figure(size=(1150, 720))

    ax_scene = Axis(fig[1:3, 1];
        xlabel="position (m)", ylabel="time (s)",
        title=title)

    block_color = (:dodgerblue3, 0.28)
    block_edge = (:dodgerblue4, 0.85)
    pusher_color = :orange3

    yspan = p.T / max(p.N - 1, 1)
    half_height = 0.32 * yspan
    for k in 1:p.N
        y0 = comp.times[k] - half_height
        y1 = comp.times[k] + half_height
        rect = Rect2f(comp.block_left[k], y0, 2 * p.block_half_width, y1 - y0)
        poly!(ax_scene, rect; color=block_color, strokecolor=block_edge, strokewidth=1)
        scatter!(ax_scene, [comp.pusher_pos[k]], [comp.times[k]];
            color=pusher_color, markersize=8)
    end

    lines!(ax_scene, comp.block_left, comp.times;
        color=:dodgerblue4, linewidth=2, label="block sides")
    lines!(ax_scene, comp.block_right, comp.times;
        color=:dodgerblue4, linewidth=2)
    lines!(ax_scene, comp.pusher_pos, comp.times;
        color=pusher_color, linewidth=2.5, label="pusher")

    scatter!(ax_scene, [comp.block_pos[1]], [comp.times[1]];
        color=:black, markersize=12, marker=:circle, label="start")
    scatter!(ax_scene, [comp.block_pos[end]], [comp.times[end]];
        color=:black, markersize=15, marker=:star5, label="target")
    axislegend(ax_scene; position=:lt)

    ax_force = Axis(fig[1, 2];
        xlabel="time (s)", ylabel="contact force (N)",
        title="Contact Force")
    lines!(ax_force, comp.interval_times, comp.contact_force;
        color=:seagreen4, linewidth=2.5)
    hlines!(ax_force, [0.0]; color=(:gray45, 0.55), linewidth=1)

    ax_gap = Axis(fig[2, 2];
        xlabel="time (s)", ylabel="contact gap (m)",
        title="Contact Gap")
    lines!(ax_gap, comp.times, comp.contact_gap;
        color=:purple4, linewidth=2.5)
    hlines!(ax_gap, [0.0]; color=(:gray45, 0.55), linewidth=1)

    ax_control = Axis(fig[3, 2];
        xlabel="time (s)", ylabel="pusher force (N)",
        title="Control Force")
    lines!(ax_control, comp.interval_times, comp.u;
        color=:firebrick3, linewidth=2.5)
    hlines!(ax_control, [0.0]; color=(:gray45, 0.55), linewidth=1)

    linkxaxes!(ax_force, ax_gap, ax_control)
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

plot_block_pushing_solution(problem, z_gurobi; title="1D Block Pushing (Gurobi)")

## Marble

using Marble

solver = Marble.Solver()
Marble.setup!(solver, problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types; verbosity = 1)
results = Marble.solve!(solver)
z_marble = Marble.z(results)

##

plot_block_pushing_solution(problem, z_marble; title="1D Block Pushing (Marble)")