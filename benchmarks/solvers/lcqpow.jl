const LCQPOW_OPTIONS = (
    use_dummy_constant = true,
    stationarity_tolerance = 1e-7,   # default ~2e-10 is too tight for the penalty method
    max_penalty_parameter = 1e10,    # default 1e8 caps out before complementarity converges
)

function benchmark_lcqpow(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("lcqpow"; problems, save, results_dir) do name
        data = Marble.mpcc_to_marble(get_mpcc(name))
        result = LCQPow.solve_qpcc_with_lcqpow(data; LCQPOW_OPTIONS..., x0 = zeros(length(data.q)))

        ok = result.converged
        x = ok ? result.x : nothing
        feas = ok ? feasibility_stats(data, x) : FAILED_STATS
        SolverResult(;
            successful = ok,
            status = string(ok ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE),
            solve_time = result.solve_time_seconds,
            x,
            feas...,
        )
    end
end
