function lcqpow_solve(name, options=(;))
    data = Marble.mpcc_to_marble(get_mpcc(name))
    result = LCQPow.solve_qpcc_with_lcqpow(data; options...)
    converged = result.converged
    x = result.x
    feas = residuals_or_failed(data, x)
    BenchmarkProblemResult(;
        converged,
        status = string(converged ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE),
        solve_time = result.solve_time_seconds,
        x,
        feas...,
    )
end
