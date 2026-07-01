using JuMP, Ipopt, MarbleBenchmarks, NLPModelsJuMP, Marble

# function minimal_gate_mpcc(; N=4, xgate=[10.0, -10.0])
#     h = 0.5  # gate half-width; gate is x ∈ [xgate-h, xgate+h]

#     ngates = length(xgate)

#     model = JuMP.Model()

#     @variable(model, x[1:N])          # 1D position
#     @variable(model, u[1:N-1])        # control input
#     @variable(model, μ[1:N, 1:ngates] >= 0)     # gate-credit increment
#     @variable(model, 0 <= s[1:N, 1:ngates])     # gate violation slack

#     @objective(model, Min,
#         sum(u[k]^2 for k in 1:N-1)
#     )

#     @constraint(model, x[1] == -2.0)                                  # start outside gate
#     @constraint(model, dyn[k=1:N-1], x[k+1] == x[k] + u[k])

#     @expression(model, λ[k=1:N, j=1:ngates], sum(μ[i, j] for i in 1:k))  # λ[k] = sum of μ[1:k]
#     @constraint(model, [j=1:ngates], λ[N, j] >= 1.0)

#     # If x ∈ [x_gate - h, x_gate + h] then we have:
#     # x[k] - (xgate + h) is negative, (xgate - h) - x[k] is negative, so s can be 0
#     # and μ can be nonzero. If x > xgate + h, then x[k] - (xgate + h) is positive, so s > 0
#     # and if x < xgate - h, then (xgate - h) - x[k] is positive, so s > 0
#     # so μ must be 0 in both cases.

#     @constraint(model, gate_ub[k=1:N, j=1:ngates], x[k] - (xgate[j] + h) <= s[k, j])
#     @constraint(model, gate_lb[k=1:N, j=1:ngates], (xgate[j] - h) - x[k] <= s[k, j])

#     # pairs   = var_con_complementarities(model, model[:μ], model[:gate_viol])
#     ind_cc1, ind_cc2, cc_types = complementarity_indices(model, (model[:μ], model[:s]))

#     return (; model, ind_cc1, ind_cc2, cc_types, inds=var_inds(model))
# end

# # Try slack_penalty = 0.0, 1.0 (works), 10.0

problem = get_problem(:minimal_gate_mpcc; N=3, xgate=[10.0, -10.0]) # Makes no progress with default params
inds = var_inds(problem.model)

## Ipopt

using Ipopt

κ = 1e-3
ipopt_model = reformulate_lie(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types; κ=κ, form=:ineq)

set_optimizer(ipopt_model, Ipopt.Optimizer)
optimize!(ipopt_model)

x_ipopt = value.(all_variables(ipopt_model))[1:num_variables(problem.model)];

x_ipopt[problem.inds[:x]]
x_ipopt[problem.inds[:s]]
objective_value(ipopt_model)

## Marble

solver = Marble.Solver()
Marble.setup!(solver, problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types;
              verbosity = 1, debug = true, max_iters=1000, prox_scale=0.1, relaxation_min=1e-8, penalty_max=1e8, relaxation_scaling=0.5, max_inner_iters_per_subproblem=1000)
results = Marble.solve!(solver)
z_marble = Marble.z(results)

# nlp = MathOptNLPModel(problem.model)
# marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

# opts = Marble.SolverOptions()
# Marble.verbosity!(opts, 1)
# Marble.max_iters!(opts, 1000)

# # Changing the initial relaxation sometimes allows it to converge?
# # Marble.relaxation_initial!(opts, 1e-5)
# Marble.debug!(opts, true)

# result = Marble.solve(marble_data; opts = opts)
# # result = Marble.solve(marble_data; opts = opts, z0=x_ipopt)
# x_marble = Marble.z(result);

# @show Marble.converged(result)
# @show Marble.solve_time_s(result)

debug_log = load_debug_log(joinpath(@__DIR__, "..", "solver_debug.jsonl"));
iters = get_iterates(debug_log, "z");

##
using Gurobi
gurobi_model = reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

set_optimizer(gurobi_model, Gurobi.Optimizer)
optimize!(gurobi_model)

x_gurobi = value.(all_variables(gurobi_model));

x_gurobi[problem.inds[:x]]
x_gurobi[problem.inds[:s]]
objective_value(gurobi_model)