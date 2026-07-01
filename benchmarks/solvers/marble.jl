const MARBLE_OPTIONS = (
    verbosity = 0,
    max_iters = 2000,
    relax_min = 1e-8,
    penalty_max = 1e9,
    relax_scaling = 0.1,
    use_relax_correction = true,
    convergence_eq_violation = 5e-8,
    convergence_ineq_violation = 5e-8,
    convergence_comp_violation = 5e-8,
)

function benchmark_marble(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("marble"; problems, save, results_dir) do name
        mpcc = get_mpcc(name)
        data = Marble.mpcc_to_marble(mpcc)
        solver = Marble.Solver()
        Marble.setup!(solver, mpcc; MARBLE_OPTIONS...)
        result = Marble.solve!(solver)

        ok = Marble.converged(result)
        x = ok ? Marble.z(result) : nothing
        feas = ok ? feasibility_stats(data, x) : FAILED_STATS
        SolverResult(;
            successful = ok,
            status = string(ok ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE),
            solve_time = Marble.solve_time_s(result),
            x,
            feas...,
        )
    end
end
