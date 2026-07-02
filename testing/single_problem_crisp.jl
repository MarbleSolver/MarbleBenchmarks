using JuMP, NLPModelsJuMP
using LinearAlgebra
using CRISP, MarbleBenchmarks

# problem = get_problem(:rocket_catch_3d; N=40, T=4.0);
# problem = get_problem(:multi_robot_coordination; N=30, T=5.0, d_min=5.0);
problem = get_problem(:quadrotor_gates_3d; N=40, T=9.0, ngates=4);

##

data = Marble.mpcc_to_marble(problem.mpcc)
result = CRISP.solve_qpcc_with_crisp(data; 
    # use_dummy_constant = true,
    # stationarity_tolerance = 1e-7,   # default ~2e-10 is too tight for the penalty method
    # max_penalty_parameter = 1e10,    # default 1e8 caps out before complementarity converges
)
