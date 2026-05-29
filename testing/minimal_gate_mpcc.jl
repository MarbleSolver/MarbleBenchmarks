using JuMP, Ipopt, MarbleBenchmarks, NLPModelsJuMP

function minimal_gate_mpcc(; N=2, xgate=0.0, slack_penalty=0.0)
    h = 0.5  # gate half-width; gate is x ∈ [xgate-h, xgate+h]

    model = JuMP.Model()

    @variable(model, x[1:N])          # 1D position
    @variable(model, u[1:N-1])        # control input
    @variable(model, μ[1:N] >= 0)     # gate-credit increment
    @variable(model, λ[1:N] >= 0)     # cumulative credit
    @variable(model, s[1:N] >= 0)     # gate violation slack

    @objective(model, Min,
        sum(u[k]^2 for k in 1:N-1)
        # + slack_penalty * (s' * s)
        #- λ[end]
        #- sum(λ[k] for k in 1:N)
        # + slack_penalty * sum(s)
    )

    @constraint(model, x[1] == -2.0)                                  # start outside gate
    @constraint(model, dyn[k=1:N-1], x[k+1] == x[k] + u[k])

    @constraint(model, λ_ic, λ[1] == μ[1])
    @constraint(model, λ_dyn[k=2:N], λ[k] == λ[k-1] + μ[k])
    @constraint(model, terminal, λ[N] == 1.0)                         # must pass gate

    # If x ∈ [x_gate - h, x_gate + h] then we have:
    # x[k] - (xgate + h) is negative, (xgate - h) - x[k] is negative, so s can be 0
    # and μ can be nonzero. If x > xgate + h, then x[k] - (xgate + h) is positive, so s > 0
    # and if x < xgate - h, then (xgate - h) - x[k] is positive, so s > 0
    # so μ must be 0 in both cases.

    @constraint(model, gate_ub[k=1:N], x[k] - (xgate + h) <= s[k])
    @constraint(model, gate_lb[k=1:N], (xgate - h) - x[k] <= s[k])

    # pairs   = var_con_complementarities(model, model[:μ], model[:gate_viol])
    pairs   = var_var_complementarities(model, model[:μ], model[:s])  # μ[k] ⟂ s[k]
    ind_cc1 = first.(pairs)
    ind_cc2 = last.(pairs)
    cc_types = fill((:var, :var), N)

    return (; model, ind_cc1, ind_cc2, cc_types, inds=var_inds(model))
end

# Try slack_penalty = 0.0, 1.0 (works), 10.0

problem = minimal_gate_mpcc(N=3, xgate=10.0, slack_penalty=1e-6) # Has linsearch failure on default params
# problem = minimal_gate_mpcc(N=2, xgate=1.0, slack_penalty=0.0) # Makes no progress with default params

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

using Marble

nlp = MathOptNLPModel(problem.model)
marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
Marble.max_iters!(opts, 1000)

# Changing the initial relaxation sometimes allows it to converge?
# Marble.relaxation_initial!(opts, 1e-5)
Marble.debug!(opts, true)

result = Marble.solve(marble_data; opts = opts)
# result = Marble.solve(marble_data; opts = opts, z0=x_ipopt)
x_marble = Marble.z(result);

@show Marble.converged(result)
@show Marble.solve_time_s(result)

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