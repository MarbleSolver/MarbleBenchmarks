using JuMP, NLPModelsJuMP
using LinearAlgebra
using MarbleBenchmarks

##

problem = get_problem(:quadrotor_gates; N=15, T=9.0, ngates=3);

using GLMakie

# Returns the 4 world-frame corners of a gate's rectangular opening,
# ordered [BL, BR, TR, TL] (bottom-left, bottom-right, top-right, top-left).
function gate_corners(pos, R, dims)
    hw, hh = dims[1]/2, dims[3]/2
    corners_gate = [[-hw, 0.0, -hh], [hw, 0.0, -hh], [hw, 0.0, hh], [-hw, 0.0, hh]]
    return Point3f.(pos + R * c for c in corners_gate)
end

function plot_quadrotor_solution(problem, x_sol)
    p = problem.params

    # Physical positions: position delta + linearization position x̄[1:3]
    x_vals = x_sol[problem.inds[:x]]       # N × 9
    pos    = x_vals[:, 1:3] .+ p.x̄[1:3]'  # N × 3

    gate_colors = [Makie.wong_colors()[i] for i in 1:p.ngates]

    fig = Figure(size=(950, 700))
    ax  = Axis3(fig[1, 1];
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        title="Quadrotor Gate Traversal",
        aspect=:data)   # equal scaling on all three axes

    for j in 1:p.ngates
        corners = gate_corners(p.gate_positions[j], p.gate_orientations[j], p.gate_dims)
        c = gate_colors[j]

        # Transparent filled quad (two triangles) for the gate opening
        verts = corners
        faces = [1 2 3; 1 3 4]
        mesh!(ax, verts, faces; color=(c, 0.3), transparency=true)

        # Opaque border frame
        border = [corners; corners[1:1]]
        GLMakie.lines!(ax, border; color=c, linewidth=3, label="Gate $j")

        # Gate label at center
        center = Point3f(p.gate_positions[j])
        GLMakie.text!(ax, center; text="Gate $j", fontsize=13, color=c,
              align=(:center, :bottom), offset=(0, 8))
    end

    # Trajectory line
    traj = [Point3f(pos[k, :]) for k in 1:p.N]
    GLMakie.lines!(ax, traj; color=:black, linewidth=2, label="Trajectory")
    GLMakie.scatter!(ax, [traj[1]];   color=:black, markersize=12, marker=:circle, label="Start")
    GLMakie.scatter!(ax, [traj[end]]; color=:black, markersize=14, marker=:star5,  label="End")

    GLMakie.Legend(fig[1, 2], ax)
    display(fig)
    return fig
end

## Gurobi

using Gurobi
gurobi_model = reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

set_optimizer(gurobi_model, Gurobi.Optimizer)

set_optimizer_attribute(gurobi_model, "MIPFocus", 1)
set_optimizer_attribute(gurobi_model, "SolutionLimit", 10)

optimize!(gurobi_model)

status = termination_status(gurobi_model)
primal = primal_status(gurobi_model)

println("Gurobi status: $(status)")
println("Gurobi primal status: $(primal)")
println("Gurobi solve time: $(solve_time(gurobi_model))")

if primal == FEASIBLE_POINT
    x_gurobi = value.(all_variables(gurobi_model));
else
    error("Gurobi did not find a feasible solution.")
end;

## Ipopt

using Ipopt

κ = 1e-1
ipopt_model = reformulate_lie(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types;
                              κ=κ, form=:eq)

set_optimizer(ipopt_model, Ipopt.Optimizer)

# set_optimizer_attribute(ipopt_model, "mu_strategy", "adaptive")
set_optimizer_attribute(ipopt_model, "tol", 1e-8)
set_optimizer_attribute(ipopt_model, "constr_viol_tol", 1e-8)
# set_optimizer_attribute(ipopt_model, "compl_inf_tol", 1e-8)
# set_optimizer_attribute(ipopt_model, "bound_push", 1e-8)
# set_optimizer_attribute(ipopt_model, "bound_frac", 1e-8)
set_optimizer_attribute(ipopt_model, "linear_solver", "ma27") # or ma57 if available

optimize!(ipopt_model)

x_ipopt = value.(all_variables(ipopt_model))[1:num_variables(problem.model)];

## Marble

using Marble

nlp = MathOptNLPModel(problem.model)
marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
Marble.max_iters!(opts, 4000)
Marble.relaxation_initial!(opts, 1e-4)
Marble.convergence_comp_violation!(opts, 1e-1)
# Marble.penalty_initial!(opts, 1.0)

result = Marble.solve(marble_data; opts = opts)

@show Marble.converged(result)
@show Marble.solve_time_s(result)

x_marble = Marble.z(result);

##
plot_quadrotor_solution(problem, x_gurobi)
plot_quadrotor_solution(problem, x_ipopt)
plot_quadrotor_solution(problem, x_marble)

plot_quadrotor_solution(problem, result.x)
