using Printf, MarbleBenchmarks, Marble

# ===========================================================================
# Summarise the latest benchmark run into a readable table (and copy-paste LaTeX
# rows). Reads the newest CSV per solver from results/stats/ — columns:
#   problem, status, converged, obj, residual_eq, residual_ineq, residual_comp, total_time_s
#
# Usage:
#   julia analysis.jl                       # all problems, in first-seen order
#   julia analysis.jl transport hopper_2d   # only these, in the order given
# ===========================================================================

const STATS_DIR = joinpath(@__DIR__, "results", "stats")

# solvers in table (row) order: CSV prefix => display name
const SOLVERS = [
    "gurobi" => "Gurobi",
    "lcqpow" => "LCQPow",
    "crisp"  => "CRISP",
    "marble" => "Marble",
]

# a solution is "feasible" if every residual is below this
const FEAS_TOL = 1e-6

# --- load the newest CSV for a solver --------------------------------------
function latest_csv(prefix)
    files = filter(f -> startswith(f, prefix * "_") && endswith(f, ".csv"), readdir(STATS_DIR))
    isempty(files) && return nothing
    return joinpath(STATS_DIR, last(sort(files)))
end

# parse a stats CSV → (order of problems, Dict(problem => NamedTuple))
function parse_csv(path)
    rows = Dict{String,NamedTuple}()
    order = String[]
    lines = readlines(path)
    header = split(strip(lines[1]), ',')
    col(f, name) = f[findfirst(==(name), header)]
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(strip(ln), ',')
        prob = col(f, "problem")
        push!(order, prob)
        rows[prob] = (
            status    = col(f, "status"),
            converged = parse(Bool, col(f, "converged")),
            obj       = parse(Float64, col(f, "obj")),
            r_eq      = parse(Float64, col(f, "residual_eq")),
            r_ineq    = parse(Float64, col(f, "residual_ineq")),
            r_comp    = parse(Float64, col(f, "residual_comp")),
            time_s    = parse(Float64, col(f, "total_time_s")),
        )
    end
    return order, rows
end

feasible(r) = max(r.r_eq, r.r_ineq, r.r_comp) <= FEAS_TOL
timed_out(r) = r.status == "TIME_LIMIT"
fmt_time(r) = timed_out(r) ? ">600" : @sprintf("%.4g", r.time_s)
fmt_obj(r)  = @sprintf("%.2f", r.obj)

# problem dimensions from the MPCC → Marble reduction (matches licq_analysis.jl):
#   n = #decision vars, m = #eq + #ineq constraints, p = #complementarity pairs
function problem_dims(prob)
    d = Marble.mpcc_to_marble(Benchmarks.get_mpcc(Symbol(prob)))
    return (; n = length(d.q), m = length(d.b_eq) + length(d.b_ineq),
            p = length(d.l), n_eq = length(d.b_eq), n_ineq = length(d.b_ineq))
end

# --- load everything --------------------------------------------------------
data = Dict{String,Union{Dict{String,NamedTuple},Nothing}}()
available = String[]                       # union of problems, in first-seen order
for (prefix, _) in SOLVERS
    path = latest_csv(prefix)
    if path === nothing
        data[prefix] = nothing
        println("# $(rpad(prefix, 7)) ← (no CSV found)")
        continue
    end
    order, rows = parse_csv(path)
    data[prefix] = rows
    for p in order
        p in available || push!(available, p)
    end
    println("# $(rpad(prefix, 7)) ← ", basename(path))
end

# problems to show: command-line args (in the given order) if any, else all found
if isempty(ARGS)
    problems = available
else
    unknown = filter(p -> p ∉ available, ARGS)
    isempty(unknown) || println("# WARNING: no data for requested problem(s): ", join(unknown, ", "))
    problems = filter(p -> p ∈ available, ARGS)
    isempty(problems) && error("none of the requested problems were found; available: $(join(available, ", "))")
end
println("# feasible ⇔ max(residual_eq, residual_ineq, residual_comp) ≤ $(FEAS_TOL)\n")

# problem dimensions (built once from the MPCC; nothing if a problem fails to build)
dims = Dict{String,Any}()
for prob in problems
    dims[prob] = try
        problem_dims(prob)
    catch err
        println("# WARNING: could not compute dims for $prob: $err")
        nothing
    end
end

dims_str(prob) = (d = dims[prob]; d === nothing ? "(dims unavailable)" :
    "n=$(d.n) vars,  m=$(d.m) (eq=$(d.n_eq) + ineq=$(d.n_ineq)),  p=$(d.p) comp")

# --- readable per-problem table ---------------------------------------------
for prob in problems
    println("="^78)
    println(prob, "   ", dims_str(prob))
    println("-"^78)
    @printf("  %-8s  %-9s  %-14s  %-11s  %s\n", "Solver", "Feasible", "Objective", "Time (s)", "Status")
    for (prefix, name) in SOLVERS
        rows = data[prefix]
        if rows === nothing || !haskey(rows, prob)
            @printf("  %-8s  %-9s  %-14s  %-11s  %s\n", name, "-", "-", "-", "no data")
            continue
        end
        r = rows[prob]
        @printf("  %-8s  %-9s  %-14s  %-11s  %s\n",
                name, feasible(r) ? "yes" : "NO",
                fmt_obj(r), fmt_time(r), r.status)
    end
    println()
end

# --- copy-paste LaTeX rows (one 3-cell block per problem) -------------------
# Feasible → \cmark, infeasible → \xmark with \na / {\na} for obj / time.
println("="^78)
println("LaTeX rows (problem order: ", join(problems, ", "), ")")
println("-"^78)
for (prefix, name) in SOLVERS
    rows = data[prefix]
    cells = String[]
    for prob in problems
        if rows === nothing || !haskey(rows, prob)
            push!(cells, "{--} & {--} & {--}")
            continue
        end
        r = rows[prob]
        if feasible(r)
            push!(cells, "\\cmark & \$$(fmt_obj(r))\$ & $(fmt_time(r))")
        else
            push!(cells, "\\xmark & \\na & {\\na}")
        end
    end
    println(rpad(name, 8), " & ", join(cells, "\n         & "), " \\\\")
end
println()
