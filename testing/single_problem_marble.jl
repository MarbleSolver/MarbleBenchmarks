using JuMP, NLPModelsJuMP
using LinearAlgebra
using Marble, MarbleBenchmarks

problem = get_problem(:rocket_catch_3d;
    N=40, T=4.0,
    x0=[0.1, 0.1, 25.0],
    q0=[1.0, 0.0, 0.0, 0.0],
    v0=[0.0, 0.0, -1.2],
    w0=[0.0, 0.0, 0.0],
);

##
solver = Marble.Solver()
Marble.setup!(solver, problem.mpcc;
    relax_scaling=0.5,
    use_relax_correction=true,
    verbosity=1,
)
result = Marble.solve!(solver)
Marble.solve_time_s(result) * 1e3