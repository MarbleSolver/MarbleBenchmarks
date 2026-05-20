using JuMP, NLPModelsJuMP
using LinearAlgebra
using MarbleBenchmarks

##

problem = get_problem(:quadrotor_gates_double_integrator; N=20, T=9.0, ngates=4);

##

using Marble

nlp = MathOptNLPModel(problem.model)
marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
Marble.max_iters!(opts, 4000)

Marble.relaxation_initial!(opts, 1e-4)
# Marble.relaxation_min!(opts, 1e-8)
Marble.penalty_max!(opts, 1e8)
# Marble.convergence_comp_violation!(opts, 1e-1)
# Marble.penalty_initial!(opts, 1.0)

result = Marble.solve(marble_data; opts = opts)

@show Marble.converged(result)
@show Marble.solve_time_s(result)

x_marble = Marble.z(result);
