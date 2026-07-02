using JuMP, NLPModelsJuMP
using LinearAlgebra
using Marble, MarbleBenchmarks

# problem = get_problem(:multi_robot_coordination; N=30, T=5.0, d_min=5.0);
problem = get_problem(:quadrotor_gates_3d; N=30, T=9.0, ngates=4);

##
solver = Marble.Solver()
Marble.setup!(solver, problem.mpcc;
    verbosity=1,
    relaxation_scaling=0.5,
    # convergence_comp_violation=1e-2,
    # use_relax_correction=false,

    # relaxation_scaling=0.5,
    # max_iters=1000,
    # # relaxation_initial=0.01,
    # # penalty_initial=100.0,
    # # penalty_max=1e6,
    # # relaxation_min=1e-8,
    # # use_relax_correction=false,
    # verbosity=1,
)
result = Marble.solve!(solver)

@info Marble.converged(result)
@info Marble.solve_time_s(result) * 1e3
@info Marble.obj(solver, Marble.z(result))