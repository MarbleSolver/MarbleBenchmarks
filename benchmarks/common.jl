default_results_dir() = joinpath(pwd(), "results")

benchmark_result_df() = DataFrame(
    problem = String[],
    status = String[],
    converged = Bool[],           # what the solver itself reported (informational)
    obj = Float64[],
    residual_eq = Float64[],      # feasibility residuals of the returned point;
    residual_ineq = Float64[],    # the analysis thresholds these to decide "solved"
    residual_comp = Float64[],
    total_time_s = Float64[],
)

# Results live under `results_dir` split into two subfolders: `stats/` holds the
# per-run summary CSVs, `solutions/` holds the raw solution vectors.
stats_dir(results_dir)     = joinpath(results_dir, "stats")
solutions_dir(results_dir) = joinpath(results_dir, "solutions")

function save_df_csv(df, results_dir, name, timestamp)
    dir = stats_dir(results_dir)
    mkpath(dir)
    CSV.write(joinpath(dir, "$(name)_$timestamp.csv"), df)
end

# One file per solver run, mapping problem name → returned solution vector (or
# `nothing` when the solver produced no point). Serialization is built in and
# round-trips Julia values directly, so `load_solutions` hands the Dict back
# as-is with no schema to maintain.
function save_solutions(solutions, results_dir, name, timestamp)
    dir = solutions_dir(results_dir)
    mkpath(dir)
    serialize(joinpath(dir, "$(name)_$timestamp.jls"), solutions)
end

load_solutions(path) = deserialize(path)

# Every problem exposes a Marble.MPCC, either directly or nested in a NamedTuple.
function get_mpcc(name; kwargs...)
    p = get_problem(name; kwargs...)
    p isa Marble.MPCC && return p
    @assert p isa NamedTuple "get_problem($name) must return a NamedTuple or a Marble.MPCC"
    idx = findfirst(v -> v isa Marble.MPCC, values(p))
    @assert idx !== nothing "get_problem($name) NamedTuple contains no Marble.MPCC"
    return values(p)[idx]
end

const FAILED_STATS = (obj = Inf, residual_eq = Inf, residual_ineq = Inf, residual_comp = Inf)

function feasibility_stats(data, x)
    (;
        obj = Marble.obj(data, x),
        residual_eq = norm(Marble.residual_eq(data, x), Inf),
        residual_ineq = norm(min.(0, Marble.residual_ineq(data, x)), Inf),
        residual_comp = norm(Marble.residual_comp(data, x), Inf),
    )
end

# Feasibility residuals of the returned point (for the CSV), or FAILED_STATS
# when there is no usable point. This only *reports* residuals - the "solved"
# decision lives in the analysis, which thresholds them.
function residuals_or_failed(data, x)
    x === nothing && return FAILED_STATS
    try
        feasibility_stats(data, x)
    catch
        FAILED_STATS
    end
end

# What every benchmark_* solve returns; each solve_one splats `feasibility_stats`
# (or FAILED_STATS) into the feasibility fields.
Base.@kwdef struct BenchmarkProblemResult
    converged::Bool               # the solver's own convergence report
    status::String
    solve_time::Float64
    x = nothing
    obj::Float64 = Inf
    residual_eq::Float64 = Inf
    residual_ineq::Float64 = Inf
    residual_comp::Float64 = Inf
end

# Shared driver. `solve_one(name, options) -> BenchmarkProblemResult` is the only
# per-solver logic; `solver_name` names the CSV and the options are forwarded to it.
function run_benchmark(solver_name, solve_one; problems, options=(;),
                       save=false, results_dir=default_results_dir())
    df = benchmark_result_df()
    solutions = Dict{String,Any}()          # problem name → returned solution vector (or nothing)
    for name in problems
        r = solve_one(name, options)
        push!(df, (
            problem = string(name),
            status = r.status,
            converged = r.converged,
            obj = r.obj,
            residual_eq = r.residual_eq,
            residual_ineq = r.residual_ineq,
            residual_comp = r.residual_comp,
            total_time_s = r.solve_time,
        ))
        solutions[string(name)] = r.x
    end
    if save
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        @info "Saving results under $results_dir (stats/ + solutions/)"
        save_df_csv(df, results_dir, solver_name, timestamp)
        save_solutions(solutions, results_dir, solver_name, timestamp)
    end
    return df
end
