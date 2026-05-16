using JuMP, NLPModelsJuMP
using LinearAlgebra
using BlockDiagonals
import ForwardDiff as FD
using ControlSystems
using MarbleBenchmarks

# Quaternion / attitude helpers

hat(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

function L(q)
    s, v = q[1], q[2:4]
    [s -v'; v s*I(3) + hat(v)]
end

const _T = Diagonal([1.0; -ones(3)])
const _H = [zeros(1, 3); I(3)]

qtoQ(q)  = _H' * _T * L(q) * _T * L(q) * _H
Gq(q)    = L(q) * _H
Eq(q)    = BlockDiagonal([1.0*I(3), Gq(q), 1.0*I(3)])
rptoq(ϕ) = (1 / sqrt(1 + ϕ'ϕ)) * [1; ϕ]
qtorp(q) = q[2:4] / q[1]

# Maps a full state [pos(3); quat(4); vel(3)] to its delta in reduced coordinates
# relative to a reference full state x̄_full. The attitude error is computed as
# the delta quaternion δq = q̄⁻¹ ⊗ q
function state_to_delta(x_full, x̄_full)
    q̄  = x̄_full[4:7]
    q   = x_full[4:7]
    δq  = L(q̄)' * q  # L(q̄)' = L(q̄*), so this is q̄⁻¹ ⊗ q
    [x_full[1:3] - x̄_full[1:3]; qtorp(δq); x_full[8:10] - x̄_full[8:10]]
end
Rz(θ)   = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1.0]

function quad_dynamics(x, u; m, g)
    q, v         = x[4:7], x[8:10]
    T_thrust, ω  = u[1], u[2:4]
    [v; 0.5*L(q)*_H*ω; [0; 0; -g] + (1/m)*qtoQ(q)*[0; 0; T_thrust]]
end

function linearized_quad_dynamics(x̄, ū, Δt; m, g)
    nx, nu = 10, 4
    q0 = x̄[4:7]
    Af = FD.jacobian(x -> quad_dynamics(x, ū; m=m, g=g), x̄)
    Bf = FD.jacobian(u -> quad_dynamics(x̄, u; m=m, g=g), ū)
    cf = quad_dynamics(x̄, ū; m=m, g=g)
    sysc = ss(Af, [Bf cf], I(nx), zeros(nx, nu + 1))
    sysd = c2d(sysc, Δt, :zoh)
    Ã = Eq(q0)' * sysd.A * Eq(q0)
    B̃ = Eq(q0)' * sysd.B
    return Ã, B̃[:, 1:nu], B̃[:, end]
end

function quadrotor_gates(; N=30, T=10.0)
    m, g   = 1.0, 9.81
    Δt     = T / (N - 1)
    nx̃     = 9  # pos(3) + reduced attitude(3) + vel(3)
    nu     = 4  # thrust(1) + angular rates(3); stored as deviation from hover
    ngates = 4

    q0 = [1.0; 0; 0; 0]
    x̄  = [zeros(3); q0; zeros(3)]
    ū  = [m*g; zeros(3)]
    Ã, B̃, c̃ = linearized_quad_dynamics(x̄, ū, Δt; m=m, g=g)

    gate_positions    = [[-2.0, 2.0, 0.25], [1.0, 6.0, 1.0], [2.0, 3.0, 2.0], [-4.0, 6.0, 0.5]]
    gate_orientations = [Rz(deg2rad(45.0)), Rz(deg2rad(-45.0)), Rz(0.0), Rz(deg2rad(20.0))]
    gate_dims         = [1.0, 0.1, 1.0]  # OBB half-extents in gate frame: x=width, y=depth (0=planar), z=height

    Tmin, Tmax = 0.0, 4.0*m*g
    ωmin, ωmax = [-2.0, -2.0, -1.5], [2.0, 2.0, 1.5]
    ϕmin = deg2rad.([-20.0, -20.0, -20.0])
    ϕmax = deg2rad.([ 20.0,  20.0,  20.0])

    # Specify start/final in full state space [pos(3); quat(4); vel(3)], then
    # compute the delta relative to the linearization point x̄.
    x0_full = [[0.0; 0.0; 0.0]; q0; zeros(3)]
    xF_full = [[0.0; 10.0; 0.0]; [1.0; 0.0; 0.0; 0.0]; zeros(3)]

    xstart = state_to_delta(x0_full, x̄)
    xfinal = state_to_delta(xF_full, x̄)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx̃])
    @variable(model, u[1:N-1, 1:nu])             # deviation from hover ū
    @variable(model, e[1:N, 1:ngates, 1:3])       # position error in gate frame
    @variable(model, λ[1:N, 1:ngates] >= 0)       # cumulative gate-completion state
    @variable(model, μ[1:N, 1:ngates] >= 0)       # per-step gate-completion increment

    # Minimize absolute control effort + gate violation penalty
    @objective(model, Min, sum(Δt/2 * (ū + u[k, :])' * (ū + u[k, :]) for k=1:N-1))

    # Boundary conditions
    @constraint(model, ic, x[1, :] .== xstart)
    @constraint(model, tc, x[N, :] .== xfinal)
    @constraint(model, λ_terminal, λ[N, :] .== 1.0)

    # Linearized quadrotor dynamics (reduced attitude state, deviation control)
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ã * x[k, :] + B̃ * u[k, :] + c̃)

    # Attitude (roll-pitch) bounds
    @constraint(model, rp_lb[k=1:N, d=1:3], x[k, 3+d] >= ϕmin[d])
    @constraint(model, rp_ub[k=1:N, d=1:3], x[k, 3+d] <= ϕmax[d])

    # Thrust and angular rate bounds (as deviations from hover)
    @constraint(model, T_lb[k=1:N-1], ū[1] + u[k, 1] >= Tmin)
    @constraint(model, T_ub[k=1:N-1], ū[1] + u[k, 1] <= Tmax)
    @constraint(model, ω_lb[k=1:N-1], ū[2:4] + u[k, 2:4] .>= ωmin)
    @constraint(model, ω_ub[k=1:N-1], ū[2:4] + u[k, 2:4] .<= ωmax)

    # Waypoint dynamics: λ[k,j] accumulates μ[k,j] at each timestep
    @constraint(model, waypoint_dyn_ic, λ[1, :] .== μ[1, :])
    @constraint(model, waypoint_dyn[k=2:N], λ[k, :] .== λ[k-1, :] + μ[k, :])

    # Enforce sequential gate ordering: λ[k,1] >= λ[k,2] >= ... >= λ[k,ngates]
    @constraint(model, waypoint_order[k=1:N, j=1:ngates-1], λ[k, j] >= λ[k, j+1])

    # Gate error: position error expressed in gate j's local frame.
    # x[k, 1:3] is the position delta; recover physical position by adding x̄[1:3].
    @expression(model, gate_error[k=1:N, j=1:ngates], gate_orientations[j]' * (x̄[1:3] + x[k, 1:3] - gate_positions[j]))

    # σ = min(0, min_d(gate_dims[d]/2 - |e[d]|))
    # σ = sum( max.(|e| - gate_dims./2, 0) )

    # If σ = 0, then the drone lies inside the gate's OBB along all dimensions

    @variable(model, e_abs[1:N, 1:ngates, 1:3])
    @variable(model, λ_e_abs[1:N, 1:ngates, 1:3] >= 0)

    ## Complementarity constraints to model e_abs = |gate_error|:
    # 0 ≤ λ ⟂ s - v ≥ 0  (con-var: λ_e_abs is the var, con_e_abs_rhs1 is the con)
    @constraint(model, con_e_abs_rhs1[k=1:N, j=1:ngates, d=1:3], e_abs[k,j,d] .- gate_error[k,j][d] >= 0)

    # 0 ≤ (1 - λ) ⟂ s + v ≥ 0  (con-con: con_lambda_ub ⟂ con_e_abs_rhs2)
    @constraint(model, con_e_abs_lhs2[k=1:N, j=1:ngates, d=1:3], (1 .- λ_e_abs[k,j,d]) >= 0)
    @constraint(model, con_e_abs_rhs2[k=1:N, j=1:ngates, d=1:3], e_abs[k,j,d] .+ gate_error[k,j][d] >= 0)

    ## Complementarity constraints to model pairwise_max_e_abs = max.(|e| - gate_dims./2, 0)
    @variable(model, pairwise_max_e_abs[1:N, 1:ngates, 1:3] >= 0)
    @variable(model, λ_pairwise_max_e_abs[1:N, 1:ngates, 1:3] >= 0)

    # 0 ≤ λ ⟂ (s - v) ≥ 0  (con-var: λ_pairwise_max_e_abs is the var, con_pairwise_max_e_abs_rhs1 is the con)
    @constraint(model, con_pairwise_max_e_abs_rhs1[k=1:N, j=1:ngates, d=1:3], pairwise_max_e_abs[k,j,d] .- (gate_dims[d]./2 .- e_abs[k,j,d]) >= 0)

    # 0 ≤ (1 - λ) ⟂ (s - 0) ≥ 0  (con-var: pairwise_max_e_abs is the var, con_pairwise_max_e_abs_lhs2 is the con)
    @constraint(model, con_pairwise_max_e_abs_lhs2[k=1:N, j=1:ngates, d=1:3], (1 - λ_pairwise_max_e_abs[k,j,d]) >= 0)

    ## Complementarity constraints to model 0 ≤ sum(pairwise_max_e_abs) ⟂ μ[k,j] ≥ 0
    @constraint(model, con_sum_pairwise_max_e_abs_lhs[k=1:N, j=1:ngates], sum(pairwise_max_e_abs[k,j,:]) >= 0)

    # --- Complementarity assembly ---
    # Each entry: (pairs_vector, type_tuple)
    complementarities = [
        # Gate completion: μ[k,j] ⟂ Σ_d max(|e[k,j,d]| - dims[d]/2, 0)
        (var_con_complementarities(model, model[:μ],                      model[:con_sum_pairwise_max_e_abs_lhs]), (:var, :con)),

        # |e| = max(e, 0) + max(-e, 0) LP complementarities
        (var_con_complementarities(model, model[:λ_e_abs],                model[:con_e_abs_rhs1]),                (:var, :con)),
        (con_con_complementarities(model, model[:con_e_abs_lhs2],         model[:con_e_abs_rhs2]),                (:con, :con)),

        # max(|e| - dims/2, 0) LP complementarities
        (var_con_complementarities(model, model[:λ_pairwise_max_e_abs],   model[:con_pairwise_max_e_abs_rhs1]),   (:var, :con)),
        (var_con_complementarities(model, model[:pairwise_max_e_abs],     model[:con_pairwise_max_e_abs_lhs2]),   (:var, :con)),
    ]

    ind_cc1  = reduce(vcat, first.(p) for (p, _) in complementarities)
    ind_cc2  = reduce(vcat, last.(p)  for (p, _) in complementarities)
    cc_types = reduce(vcat, fill(t, length(p)) for (p, t) in complementarities)

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, Δt, x̄, ū, ngates, gate_positions, gate_orientations, gate_dims))
end

problem = quadrotor_gates()
nvar = num_variables(problem.model)

## Gurobi

using Gurobi
gurobi_model = reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

set_optimizer(gurobi_model, Gurobi.Optimizer)
set_silent(gurobi_model)
optimize!(gurobi_model)

status = termination_status(gurobi_model)
@show status
@show solve_time(gurobi_model)

x_gurobi = value.(all_variables(gurobi_model));

## CCOpt

using CCOpt, MadNLP, MadNLPHSL

to_ccopt_cctype(x) = Dict(
    (:var, :var) => CCOpt.VarVar,
    (:var, :con) => CCOpt.VarCon,
    (:con, :var) => CCOpt.ConVar,
    (:con, :con) => CCOpt.ConCon,
)[x]

nlp = MathOptNLPModel(problem.model)
ind_cc1_cp = copy(problem.ind_cc1)
ind_cc2_cp = copy(problem.ind_cc2)

mpcc = CCOpt.MPCCModel(nlp, ind_cc1_cp, ind_cc2_cp, to_ccopt_cctype.(problem.cc_types))
vmpcc = CCOpt.vertical_form(mpcc)

opts = CCOpt.RelaxationOptions()
madnlpc_solver_options = Dict(
    :bound_relax_factor=>0.0,
    :bound_push=>1e-1,
    :print_level=>MadNLP.ERROR,
    :max_iter=>4000,
    :linear_solver=>Ma27Solver,
)

solver = CCOpt.RelaxationSolver(vmpcc; solver_opts=opts, madnlpc_solver_options...)
stats = CCOpt.solve_homotopy!(solver)

@show stats.status
@show stats.counters.total_time 

x_ccopt = stats.solution[1:nvar];

## Marble

using Marble

nlp = MathOptNLPModel(problem.model)
marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
Marble.max_iters!(opts, 4000)
# Marble.relaxation_initial!(opts, 0.01)
# Marble.penalty_initial!(opts, 1.0)

result = Marble.solve(marble_data; opts = opts)

@show Marble.converged(result)
@show Marble.solve_time_s(result)

## Plot solution

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

if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    nvar  = num_variables(problem.model)
    x_sol = value.(all_variables(gurobi_model))[1:nvar]
    plot_quadrotor_solution(problem, x_sol)
end

