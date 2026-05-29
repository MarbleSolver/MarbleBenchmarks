using JuMP, NLPModelsJuMP
using LinearAlgebra
using MarbleBenchmarks

##

problem = get_problem(:quadrotor_gates; N=40, T=9.0, ngates=2);

##
using GLMakie

function gate_corners(pos, R, dims)
    hw, hh = dims[1] / 2, dims[3] / 2
    corners_gate = [[-hw, 0.0, -hh], [hw, 0.0, -hh], [hw, 0.0, hh], [-hw, 0.0, hh]]
    return Point3f.(pos + R * c for c in corners_gate)
end

function plot_quadrotor_solution(problem, z; title="Quadrotor Gate Traversal")
    p = problem.params
    x_vals = z[problem.inds[:x]]
    pos = x_vals[:, 1:3] .+ p.x̄[1:3]'

    gate_colors = [Makie.wong_colors()[i] for i in 1:p.ngates]

    fig = Figure(size=(950, 700))
    ax = Axis3(fig[1, 1];
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        title=title,
        aspect=:data)

    for j in 1:p.ngates
        corners = gate_corners(p.gate_positions[j], p.gate_orientations[j], p.gate_dims)
        c = gate_colors[j]
        mesh!(ax, corners, [1 2 3; 1 3 4]; color=(c, 0.3), transparency=true)
        lines!(ax, [corners; corners[1:1]]; color=c, linewidth=3, label="Gate $j")
        text!(ax, Point3f(p.gate_positions[j]);
            text="Gate $j", fontsize=13, color=c, align=(:center, :bottom), offset=(0, 8))
    end

    traj = [Point3f(pos[k, :]) for k in 1:p.N]
    lines!(ax, traj; color=:black, linewidth=2, label="Trajectory")
    scatter!(ax, [traj[1]];   color=:black, markersize=12, marker=:circle, label="Start")
    scatter!(ax, [traj[end]]; color=:black, markersize=14, marker=:star5,  label="End")

    Legend(fig[1, 2], ax)
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

plot_quadrotor_solution(problem, z_gurobi; title="Quadrotor Gate Traversal (Gurobi)")

## Ipopt

using Ipopt

κ = 1e-2
ipopt_model = reformulate_lie(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types;
                              κ=κ, form=:ineq)
set_optimizer(ipopt_model, Ipopt.Optimizer)
optimize!(ipopt_model)

status = termination_status(ipopt_model)
primal  = primal_status(ipopt_model)

println("Ipopt status: $(status)")
println("Ipopt primal status: $(primal)")
println("Ipopt solve time: $(solve_time(ipopt_model))")

if primal == FEASIBLE_POINT
    z_ipopt = value.(all_variables(ipopt_model))[1:num_variables(problem.model)]
else
    error("Ipopt did not find a feasible solution.")
end

##

plot_quadrotor_solution(problem, z_ipopt; title="Quadrotor Gate Traversal (Ipopt)")

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

plot_quadrotor_solution(problem, z_marble; title="Quadrotor Gate Traversal (Marble)")
