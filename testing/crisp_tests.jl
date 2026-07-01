using Marble, MarbleBenchmarks
using CRISP

name = Symbol("liswet1-200")
mpcc = get_problem(name)
data = Marble.mpcc_to_marble(mpcc)

obj(x) = 0.5 * x' * marble.Q * x + marble.q' * x + marble.c0

result = CRISP.solve_qpcc_with_crisp(marble.Q, marble.q, marble.c0;
    J_eq = marble.J_eq,
    b_eq = marble.b_eq,
    J_ineq = marble.J_ineq,
    b_ineq = marble.b_ineq,
    L = marble.L,
    l = marble.l,
    R = marble.R,
    r = marble.r,
)

@show obj(result.x)
@show result.converged
@show result.x
@show result.solve_time_seconds
@show result.qp_time_seconds
@show result.iterations
