default_results_dir() = joinpath(pwd(), "results")

results_df() = DataFrame(
    problem = String[],
    status = String[],
    obj = Float64[],
    residual_eq = Float64[],
    residual_ineq = Float64[],
    residual_comp = Float64[],
    total_time_s = Float64[],
)

function save_df_csv(df, results_dir, name)
    mkpath(results_dir)
    dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    CSV.write(joinpath(results_dir, "$(name)_$dt.csv"), df)
end

# Every problem exposes a Marble.MPCC, either directly or nested in a NamedTuple.
function get_mpcc(name)
    p = get_problem(name)
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

# What every benchmark_* solve returns; each solve_one splats `feasibility_stats`
# (or FAILED_STATS) into the feasibility fields.
Base.@kwdef struct SolverResult
    successful::Bool
    status::String
    solve_time::Float64
    x = nothing
    obj::Float64 = Inf
    residual_eq::Float64 = Inf
    residual_ineq::Float64 = Inf
    residual_comp::Float64 = Inf
end

# Shared driver: `solve_one(name) -> SolverResult` is the only per-solver logic.
function run_benchmark(solve_one, csv_name; problems, save=false, results_dir=default_results_dir())
    df = results_df()
    for name in problems
        r = solve_one(name)
        push!(df, (
            problem = string(name),
            status = r.status,
            obj = r.obj,
            residual_eq = r.residual_eq,
            residual_ineq = r.residual_ineq,
            residual_comp = r.residual_comp,
            total_time_s = r.successful ? r.solve_time : Inf,
        ))
    end
    if save
        @info "Saving results to CSV in directory: $results_dir"
        save_df_csv(df, results_dir, csv_name)
    end
    return df
end
