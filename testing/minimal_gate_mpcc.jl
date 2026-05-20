using JuMP, MarbleBenchmarks, NLPModelsJuMP

function minimal_gate_mpcc_minimal_s(; N=2, xgate=0.0)
    h = 0.5
    L = xgate - h
    U = xgate + h

    model = JuMP.Model()

    @variable(model, x[1:N])
    @variable(model, u[1:N-1])
    @variable(model, μ[1:N] >= 0)
    @variable(model, λ[1:N] >= 0)

    # Actual gate-violation slack, now forced to be minimal via KKT conditions
    @variable(model, s[1:N] >= 0)

    # Margins for the two epigraph constraints:
    #
    # eta_ub = s - (x - U) >= 0
    # eta_lb = s - (L - x) >= 0
    @variable(model, eta_ub[1:N] >= 0)
    @variable(model, eta_lb[1:N] >= 0)

    # KKT multipliers for the hidden problem:
    #
    # min_s s
    # s.t. s >= 0
    #      s >= x - U
    #      s >= L - x
    @variable(model, alpha0[1:N] >= 0)
    @variable(model, alpha_ub[1:N] >= 0)
    @variable(model, alpha_lb[1:N] >= 0)

    @objective(model, Min,
        sum(u[k]^2 for k in 1:N-1)
    )

    @constraint(model, x[1] == -2.0)
    @constraint(model, dyn[k=1:N-1], x[k+1] == x[k] + u[k])

    @constraint(model, λ_ic, λ[1] == μ[1])
    @constraint(model, λ_dyn[k=2:N], λ[k] == λ[k-1] + μ[k])
    @constraint(model, terminal, λ[N] == 1.0)

    # Gate violation epigraph, written using explicit nonnegative margins
    @constraint(model, gate_ub_margin[k=1:N],
        eta_ub[k] == s[k] - (x[k] - U)
    )

    @constraint(model, gate_lb_margin[k=1:N],
        eta_lb[k] == s[k] - (L - x[k])
    )

    # Stationarity condition of hidden min_s problem:
    #
    # L = s - alpha0*s - alpha_ub*(s - (x-U)) - alpha_lb*(s - (L-x))
    #
    # dL/ds = 1 - alpha0 - alpha_ub - alpha_lb = 0
    @constraint(model, slack_min_stationarity[k=1:N],
        alpha0[k] + alpha_ub[k] + alpha_lb[k] == 1.0
    )

    # Original gate logic:
    #
    # μ[k] ⟂ s[k]
    pairs_gate = var_var_complementarities(model, model[:μ], model[:s])

    # Minimal-slack KKT complementarities:
    #
    # alpha0[k]    ⟂ s[k]
    # alpha_ub[k] ⟂ eta_ub[k]
    # alpha_lb[k] ⟂ eta_lb[k]
    pairs_min0 = var_var_complementarities(model, model[:alpha0], model[:s])
    pairs_minu = var_var_complementarities(model, model[:alpha_ub], model[:eta_ub])
    pairs_minl = var_var_complementarities(model, model[:alpha_lb], model[:eta_lb])

    pairs = vcat(pairs_gate, pairs_min0, pairs_minu, pairs_minl)

    ind_cc1 = first.(pairs)
    ind_cc2 = last.(pairs)
    cc_types = fill((:var, :var), length(pairs))

    return (; model, ind_cc1, ind_cc2, cc_types, inds=var_inds(model))
end

# Try slack_penalty = 0.0, 1.0 (works), 10.0

# problem = minimal_gate_mpcc(N=2, xgate=1.0, slack_penalty=0.0) # Has linsearch failure on default params
problem = minimal_gate_mpcc_minimal_s(N=30, xgate=10.0) # Makes no progress with default params

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

# Changing the initial relaxation sometimes allows it to converge?
# Marble.relaxation_initial!(opts, 1e-3)
Marble.debug!(opts, true)

# result = Marble.solve(marble_data; opts = opts)
result = Marble.solve(marble_data; opts = opts);
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