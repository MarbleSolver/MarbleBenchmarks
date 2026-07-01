using JuMP, Ipopt, MarbleBenchmarks, NLPModelsJuMP, Marble

problem = get_problem(Symbol("liswet1-200"))

solver = Marble.Solver()
Marble.setup!(solver, problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types;
              verbosity = 1, debug = true, max_iters=200, prox_scale=0.0, relaxation_min=1e-9, penalty_max=1e8, relaxation_scaling=0.1, max_inner_iters_per_subproblem=20, feas_tol_min=1e-6)
results = Marble.solve!(solver)
z_marble = Marble.z(results)