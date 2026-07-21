using MarbleBenchmarks, Marble

function solution_path(solver::String)
    dir = joinpath(@__DIR__, "solutions")
    files = filter(readdir(dir)) do f
        endswith(f, ".jls") && String(split(f, '_')[1]) == string(solver)
    end
    isempty(files) && error("no solution files for solver `$solver` in $dir")
    return joinpath(dir, last(sort(files)))
end

##

for solver in ["lcqpow", "gurobi", "crisp", "marble"]
    solutions = Benchmarks.load_solutions(solution_path(solver))
    problems = keys(solutions)

    println("Analyzing MPCC-LICQ for solver $solver on $(length(problems)) problems...")
    for problem in problems
        mpcc = Benchmarks.get_mpcc(Symbol(problem))
        data = Marble.mpcc_to_marble(mpcc)

        println("  $problem: ")
        mpcc_licq(data, solutions[problem]; verbose=true, τ=1e-8)
    end
end

##
solutions = Benchmarks.load_solutions(solution_path("marble"))
problems = keys(solutions)

for p in problems
    mpcc = Benchmarks.get_mpcc(Symbol(p))
    data = Marble.mpcc_to_marble(mpcc)
    println("$p: n_var = $(length(data.q)), n_eq = $(length(data.b_eq)), n_ineq = $(length(data.b_ineq)), n_comp = $(length(data.l))")
end