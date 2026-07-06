const MARBLE_OPTIONS = (
    verbosity = 0,
    max_iters = 2000,
    relaxation_min = 1e-9,
    penalty_max = 1e7,
    convergence_eq_violation = 1e-8,
    convergence_ineq_violation = 1e-8,
    convergence_comp_violation = 1e-8,
    inertia_warmstart = true,
)

function benchmark_marble(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("marble"; problems, save, results_dir) do name
        mpcc = get_mpcc(name)
        data = Marble.mpcc_to_marble(mpcc)
        solver = Marble.Solver()
        Marble.setup!(solver, mpcc; MARBLE_OPTIONS...)
        result = Marble.solve!(solver)

        # Report only: the solver's convergence flag and the residuals of the
        # returned iterate. The analysis decides "solved" from the residuals.
        converged = Marble.converged(result)
        x = Marble.z(result)
        feas = residuals_or_failed(data, x)
        SolverResult(;
            converged,
            status = string(converged ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE),
            solve_time = Marble.solve_time_s(result),
            x,
            feas...,
        )
    end
end
