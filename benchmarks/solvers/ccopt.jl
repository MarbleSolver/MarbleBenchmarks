const MADNLP_STATUS = Dict(s => string(s) for s in instances(MadNLP.Status))

const CCOPT_MADNLP_OPTIONS = (
    bound_relax_factor = 0.0,
    bound_push = 1e-1,
    print_level = MadNLP.ERROR,
    max_iter = 4000,
    linear_solver = Ma27Solver,
)

function benchmark_ccopt(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("ccopt"; problems, save, results_dir) do name
        mpcc = get_mpcc(name)
        data = Marble.mpcc_to_marble(mpcc)
        nlp = mpcc.nlp

        vmpcc = CCOpt.vertical_form(CCOpt.MPCCModelVarCon(nlp, copy(mpcc.ind_cc1), copy(mpcc.ind_cc2)))
        solver = CCOpt.RelaxationSolver(vmpcc; solver_opts=CCOpt.RelaxationOptions(), CCOPT_MADNLP_OPTIONS...)
        stats = CCOpt.solve_homotopy!(solver)

        ok = Int(stats.status) <= 2
        x = ok ? stats.solution[1:nlp.meta.nvar] : nothing
        feas = ok ? feasibility_stats(data, x) : FAILED_STATS
        SolverResult(;
            successful = ok,
            status = MADNLP_STATUS[stats.status],
            solve_time = stats.counters.total_time,
            x,
            feas...,
        )
    end
end
