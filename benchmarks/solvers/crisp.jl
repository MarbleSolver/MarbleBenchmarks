const CRISP_OPTIONS = (;)

function benchmark_crisp(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("crisp"; problems, save, results_dir) do name
        data = Marble.mpcc_to_marble(get_mpcc(name))
        result = CRISP.solve_qpcc_with_crisp(data; CRISP_OPTIONS..., x0 = zeros(length(data.q)))

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
