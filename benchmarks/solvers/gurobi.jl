# One Gurobi environment shared across all solves. It MUST be created at runtime
# in __init__, not as a precompile-time const: a Gurobi.Env baked into the
# package's precompile image holds a stale handle in a fresh session, which
# surfaces as "Gurobi Error 10002" the first time an Optimizer is instantiated.
const GRB_ENV_REF = Ref{Gurobi.Env}()

function __init__()
    try
        GRB_ENV_REF[] = redirect_stdout(devnull) do
            Gurobi.Env()
        end
    catch err
        @warn "Could not initialize Gurobi; benchmark_gurobi will be unavailable." exception = err
    end
    return
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

        set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV_REF[]))
        set_silent(model)
        for (k, v) in pairs(GUROBI_OPTIONS)
            set_attribute(model, String(k), v)
        end
        optimize!(model)

        # Report only: Gurobi's termination status / convergence and the
        # residuals of the returned point. The analysis decides "solved" from
        # the residuals. `has_values` guards against a model that finished
        # without a primal solution (where `value.` would throw).
        status = termination_status(model)
        converged = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        x = has_values(model) ? value.(model[:x]) : nothing
        feas = residuals_or_failed(data, x)
        SolverResult(;
            converged,
            status = string(status),
            solve_time = solve_time(model),
            x,
            feas...,
        )
    end
end
