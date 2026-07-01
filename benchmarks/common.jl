using Dates

function save_df_csv(df, results_dir, name)
    mkpath(results_dir)
    dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    CSV.write(joinpath(results_dir, "$(name)_$dt.csv"), df)
end

function solution_stats(problem, x; solve_successful=true)
    objective_value = solve_successful ? Marble.obj(problem, x) : Inf
    residual_eq = solve_successful ? norm(Marble.residual_eq(problem, x), Inf) : Inf
    residual_ineq = solve_successful ? norm(min.(0, Marble.residual_ineq(problem, x)), Inf) : Inf
    residual_comp = solve_successful ? norm(Marble.residual_comp(problem, x), Inf) : Inf
    return (; objective_value, residual_eq, residual_ineq, residual_comp) 
end