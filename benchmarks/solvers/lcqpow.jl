const LCQPOW_OPTIONS = (
    use_dummy_constant = true,
    stationarity_tolerance = 1e-7,   # default ~2e-10 is too tight
    max_penalty_parameter = 1e10,    # default 1e8 caps out before complementarity converges
    qp_solver = 1,                   # qpOASES sparse
)

function benchmark_lcqpow(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("lcqpow"; problems, save, results_dir) do name
        data = Marble.mpcc_to_marble(get_mpcc(name))
        result = LCQPow.solve_qpcc_with_lcqpow(data; LCQPOW_OPTIONS...)

        # Report only: the solver's convergence flag and the residuals of the
        # returned point. The analysis decides "solved" from the residuals.
        converged = result.converged
        x = result.x
        feas = residuals_or_failed(data, x)
        SolverResult(;
            converged,
            status = string(converged ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE),
            solve_time = result.solve_time_seconds,
            x,
            feas...,
        )
    end
end
