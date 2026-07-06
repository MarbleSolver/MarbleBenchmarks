const CRISP_OPTIONS = (;
    trail_tol=1e-3,
    trust_region_tol=1e-3,
    weighted_mode=1.0,
    constraint_tol=5e-8,
)

function benchmark_crisp(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("crisp"; problems, save, results_dir) do name
        data = Marble.mpcc_to_marble(get_mpcc(name))
        result = CRISP.solve_qpcc_with_crisp(data; CRISP_OPTIONS...,)

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
