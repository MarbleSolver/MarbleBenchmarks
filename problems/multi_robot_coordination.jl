using JuMP, NLPModelsJuMP
using Combinatorics
using MarbleBenchmarks

function multi_robot_coordination3(; N=30, T=5.0)
    Δt = T / (N - 1)
    m = 1.0
    d_min = 5.0

    n_robots = 3

    x0 = [
        Float64[-5, 0, 0, 0],
        Float64[0, 0, 0, 0],
        Float64[5, 0, 0, 0]
    ]

    n_slacks = binomial(n_robots, 2)

    xF = [
        Float64[0, 10, 0, 0],
        Float64[5, 10, 0, 0],
        Float64[-5, 10, 0, 0]
    ]

    model = JuMP.Model()

    @variable(model, x[1:N, 1:n_robots, 1:4])
    @variable(model, u[1:N, 1:n_robots, 1:2])

    @variable(model, s[1:N, 1:n_slacks, 1:2])
    @variable(model, λ[1:N, 1:n_slacks, 1:2] >= 0)

    @objective(model, Min, sum(Δt * 0.5 * (u[k, i, :]' * u[k, i, :]) for i in 1:n_robots, k in 1:N-1))

    @constraint(model, ic[i=1:n_robots], x[1, i, :] .== x0[i])
    @constraint(model, fc[i=1:n_robots], x[N, i, :] .== xF[i])

    continuous_dynamics(x, u; m) = [x[3:4]; u ./ m]
    @constraint(model, dynamics[k=1:N-1, i=1:n_robots], x[k+1, i, :] .== x[k, i, :] + Δt * continuous_dynamics(x[k+1, i, :], u[k, i, :]; m=m))

    pairs = collect(combinations(1:n_robots, 2))
    @expression(model, d[k=1:N, i=1:n_slacks, j=1:2], begin
        p1, p2 = pairs[i]
        x[k, p1, j] - x[k, p2, j]
    end)

    # 0 ≤ λ ⟂ s - d ≥ 0  (con-var: λ is the var, con_s_minus_d is the con)
    @constraint(model, con_s_minus_d[k=1:N, i=1:n_slacks, j=1:2], s[k, i, j] - d[k, i, j] >= 0)

    # 0 ≤ (1 - λ) ⟂ s + d ≥ 0  (con-con: con_lambda_ub ⟂ con_s_plus_d)
    @constraint(model, con_lambda_ub[k=1:N, i=1:n_slacks, j=1:2], (1 - λ[k, i, j]) >= 0)
    @constraint(model, con_s_plus_d[k=1:N, i=1:n_slacks, j=1:2], s[k, i, j] + d[k, i, j] >= 0)

    @constraint(model, [k=1:N, i=1:n_slacks], sum(s[k, i, j] for j=1:2) >= d_min)

    # con-var: λ[k,i,j] ⟂ (s - d)[k,i,j] >= 0
    var_con_pairs = var_con_complementarities(model, model[:λ], model[:con_s_minus_d])

    # con-con: (1 - λ)[k,i,j] >= 0  ⟂  (s + d)[k,i,j] >= 0
    con_con_pairs = con_con_complementarities(model, model[:con_lambda_ub], model[:con_s_plus_d])

    ind_cc1 = [first.(var_con_pairs); first.(con_con_pairs)]
    ind_cc2 = [last.(var_con_pairs); last.(con_con_pairs)]
    cc_types   = [fill((:var, :con), length(var_con_pairs)); fill((:con, :con), length(con_con_pairs))]
    
    return (; model, ind_cc1, ind_cc2, cc_types,
            inds=var_inds(model),
            params=(; N, T, d_min, n_robots, pairs=collect(combinations(1:n_robots, 2))))
end

problem = multi_robot_coordination3()

## Gurobi
using Gurobi
nvar = num_variables(problem.model)

gurobi_model = reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

set_optimizer(gurobi_model, Gurobi.Optimizer)
set_silent(gurobi_model)
optimize!(gurobi_model)

status = termination_status(gurobi_model)
x_sol = value.(all_variables(gurobi_model))[1:nvar]

@show solve_time(gurobi_model)

## Marble
using Marble

# Build model
nlp = MathOptNLPModel(problem.model)

marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 0)
Marble.relaxation_scaling!(opts, 0.1)

result = Marble.solve(marble_data; opts = opts)
x_sol = Marble.z(result)

@show Marble.solve_time_s(result)
@show Marble.converged(result)

## CCOpt

using CCOpt, MadNLP

nlp = MathOptNLPModel(problem.model)

to_ccopt_cctype(x) = Dict(
    (:var, :var) => CCOpt.VarVar,
    (:var, :con) => CCOpt.VarCon,
    (:con, :var) => CCOpt.ConVar,
    (:con, :con) => CCOpt.ConCon,
)[x]

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
    # :linear_solver=>Mumps,
)

solver = CCOpt.RelaxationSolver(vmpcc; solver_opts=opts, madnlpc_solver_options...)
stats = CCOpt.solve_homotopy!(solver)

x_sol = stats.solution[1:nlp.meta.nvar] # Extract original variables, this is safe / LiftedNLPModel enforces this

@show stats.counters.total_time
## Plot a solution
using Plots

if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    robot_sols = [x_sol[problem.inds[:x][:, i, :]] for i in 1:problem.params.n_robots]
    robot_names = ["Robot $('A' + i - 1)" for i in 1:problem.params.n_robots]
    colors = [:steelblue :coral :mediumseagreen :mediumpurple :goldenrod][:, 1:problem.params.n_robots]
    t = range(0, problem.params.T, length=problem.params.N)

    # Trajectory plot
    p1 = Plots.plot(;
        xlabel="x", ylabel="y", title="Trajectories",
        framestyle=:box, grid=true, gridalpha=0.15,
        legend=:topleft, aspect_ratio=:equal)
    for (i, (xs, name)) in enumerate(zip(robot_sols, robot_names))
        Plots.plot!(p1, xs[:, 1], xs[:, 2]; label=name, color=colors[i], lw=2)
        Plots.scatter!(p1, [xs[1, 1]], [xs[1, 2]];   color=colors[i], marker=:circle, ms=6, label=false)
        Plots.scatter!(p1, [xs[end, 1]], [xs[end, 2]]; color=colors[i], marker=:star5,  ms=8, label=false)
    end

    # L1 pairwise distance plot
    p2 = Plots.plot(;
        xlabel="time", ylabel="L1 distance", title="Pairwise L1 Distances",
        framestyle=:box, grid=true, gridalpha=0.15, legend=:topright)
    pair_colors = [:royalblue :tomato :mediumseagreen :darkorange :mediumpurple :teal]
    for (idx, (p1r, p2r)) in enumerate(problem.params.pairs)
        l1 = vec(sum(abs.(robot_sols[p1r][:, 1:2] - robot_sols[p2r][:, 1:2]), dims=2))
        name = "$(robot_names[p1r][end])–$(robot_names[p2r][end])"
        Plots.plot!(p2, t, l1; label=name, color=pair_colors[idx], lw=2)
    end
    Plots.hline!(p2, [problem.params.d_min]; linestyle=:dash, color=:black, alpha=0.4, label="d_min")

    Plots.plot(p1, p2; layout=(1, 2), size=(1050, 440), margin=5Plots.mm, dpi=150)
end