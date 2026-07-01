const GRB_ENV = redirect_stdout(devnull) do
    Gurobi.Env()
end

const GUROBI_OPTIONS = (
    # NodeLimit = 1000,
    MIPGap = 1e-4,
    FeasibilityTol = 1e-6,
)

function benchmark_gurobi(; problems, save=false, results_dir=default_results_dir())
    run_benchmark("gurobi"; problems, save, results_dir) do name
        mpcc = get_mpcc(name)
        data = Marble.mpcc_to_marble(mpcc)
        model = mpcc_to_sos1(mpcc)

        set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
        set_silent(model)
        for (k, v) in pairs(GUROBI_OPTIONS)
            set_attribute(model, String(k), v)
        end
        optimize!(model)

        status = termination_status(model)
        ok = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        x = ok ? value.(model[:x]) : nothing
        feas = ok ? feasibility_stats(data, x) : FAILED_STATS
        SolverResult(;
            successful = ok,
            status = string(status),
            solve_time = solve_time(model),
            x,
            feas...,
        )
    end
end
