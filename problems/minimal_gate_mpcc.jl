using JuMP, Ipopt, MarbleBenchmarks, NLPModelsJuMP

# Minimal "pass through gate" MPCC
# ──────────────────────────────────────────────────────────────────────────────
# 1D position x[k], N timesteps. Gate at x ∈ [-h, h] with violation slack s[k].
# μ[k]: gate-credit at step k. Complementarity: μ[k] ⟂ s[k]
#   → if s[k] > 0 (drone outside gate) then μ[k] = 0
#   → if μ[k] > 0 (drone earns credit)  then s[k] = 0
# λ[N] = 1 forces the drone to earn exactly one unit of credit total.
#
# WITHOUT slack penalty:
#   No cost on s[k] → solver keeps s large → μ ≈ 0 everywhere
#   → λ[N] = 1 becomes infeasible (or Ipopt gets stuck at a bad point)
#
# WITH slack penalty:
#   Cost drives s → 0 → drone must enter gate → μ can be nonzero → λ[N] = 1 ✓
# ──────────────────────────────────────────────────────────────────────────────

function minimal_gate_mpcc(; N=2, xgate=0.0)
    h = 0.5  # gate half-width; gate is x ∈ [xgate-h, xgate+h]

    model = JuMP.Model()

    @variable(model, x[1:N])          # 1D position
    @variable(model, u[1:N-1])        # control input
    @variable(model, μ[1:N] >= 0)     # gate-credit increment
    @variable(model, λ[1:N] >= 0)     # cumulative credit
    @variable(model, 0 <= s[1:N])     # gate violation slack

    @objective(model, Min,
        sum(u[k]^2 for k in 1:N-1)
        # + s' * s
        #- λ[end]
        #- sum(λ[k] for k in 1:N)
        #+ slack_penalty * sum(s)
    )

    @constraint(model, x[1] == -2.0)                                  # start outside gate
    @constraint(model, dyn[k=1:N-1], x[k+1] == x[k] + u[k])

    @constraint(model, λ_ic, λ[1] == μ[1])
    @constraint(model, λ_dyn[k=2:N], λ[k] == λ[k-1] + μ[k])
    @constraint(model, terminal, λ[N] == 1.0)                         # must pass gate

    # s[k] absorbs any violation of x[k] ∈ [xgate-h, xgate+h]
    @constraint(model, gate_ub[k=1:N],   x[k] - (xgate + h) <= s[k])
    @constraint(model, gate_lb[k=1:N],  (xgate - h) - x[k]  <= s[k])
    # @constraint(model, gate_viol[k=1:N], s[k] >= 0)                   # CC partner for μ

    # pairs   = var_con_complementarities(model, model[:μ], model[:gate_viol])
    pairs   = var_var_complementarities(model, model[:μ], model[:s])  # μ[k] ⟂ s[k]
    ind_cc1 = first.(pairs)
    ind_cc2 = last.(pairs)
    cc_types = fill((:var, :var), N)

    return (; model, ind_cc1, ind_cc2, cc_types, inds=var_inds(model))
end

problem = minimal_gate_mpcc(N=10, xgate=1.0)

## Ipopt

using Ipopt

κ = 1e-2
ipopt_model = reformulate_lie(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types; κ=κ, form=:eq)

set_optimizer(ipopt_model, Ipopt.Optimizer)
optimize!(ipopt_model)

x_ipopt = value.(all_variables(ipopt_model))[1:num_variables(problem.model)];

## Marble

using Marble

nlp = MathOptNLPModel(problem.model)
marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
Marble.max_iters!(opts, 1000)
Marble.relaxation_initial!(opts, 1e-2)
Marble.debug!(opts, true)

# Marble.convergence_comp_violation!(opts, 1e-)

# Marble.relaxation_initial!(opts, 0.01)
# Marble.penalty_initial!(opts, 1.0)

result = Marble.solve(marble_data; opts = opts)
# result = Marble.solve(marble_data; opts = opts, initial_z=x_ipopt)
x_marble = Marble.z(result);

@show Marble.converged(result)
@show Marble.solve_time_s(result)

debug_log = load_debug_log("/Users/micahreich/Documents/MarbleBenchmarks/solver_debug.jsonl");
iters = get_iterates(debug_log, "z");

##
using Gurobi
gurobi_model = reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

set_optimizer(gurobi_model, Gurobi.Optimizer)
optimize!(gurobi_model)

x_gurobi = value.(all_variables(gurobi_model));

##

# for penalty in [0.0, 1.0]
#     prob = minimal_gate_mpcc(; N=6, slack_penalty=penalty)
#     m = reformulate_lie(prob.model, prob.ind_cc1, prob.ind_cc2, prob.cc_types; κ=0.1, form=:eq)

#     set_optimizer(m, Ipopt.Optimizer)
#     set_silent(m)
#     optimize!(m)

#     status = termination_status(m)
#     x_val  = value.(m[:x])
#     s_val  = value.(m[:s])
#     μ_val  = value.(m[:μ])
#     λ_val  = value.(m[:λ])

#     println("\n── slack_penalty = $penalty ──")
#     println("  status : $status")
#     println("  x      : ", round.(x_val, digits=3))
#     println("  s      : ", round.(s_val, digits=3))
#     println("  μ      : ", round.(μ_val, digits=3))
#     println("  λ[end] : ", round(λ_val[end], digits=3))
# end
