GRB_ENV = redirect_stdout(devnull) do
    Gurobi.Env()
end

function benchmark_gurobi(; problems, save=false, results_dir=joinpath(@__DIR__, "results"))
    df = DataFrame(
        problem=String[],
        status=String[],
        obj = Float64[],
        residual_eq = Float64[],
        residual_ineq = Float64[],
        residual_comp = Float64[],
        total_time_s = Float64[],
    )

    for name in problems
        problem = get_problem(name)

        marble_data = MarbleData(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)
        jump_model = to_jump(marble_data)

        set_optimizer(jump_model, () -> Gurobi.Optimizer(GRB_ENV))
        set_silent(jump_model)
        optimize!(jump_model)

        status = termination_status(jump_model)

        # Compute residuals and objective value
        solve_successful = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

        if solve_successful
            x = value.(all_variables(jump_model))[1:problem.model.meta.nvar]
        end

        objective_value = solve_successful ? Marble.obj(marble_data, x) : Inf
        residual_eq = solve_successful ? norm(Marble.residual_eq(marble_data, x), Inf) : Inf
        residual_ineq = solve_successful ? norm(Marble.residual_ineq(marble_data, x), Inf) : Inf
        residual_comp = solve_successful ? norm(Marble.residual_comp(marble_data, x), Inf) : Inf

        # Add entry to df
        push!(df, (
            problem=string(name),
            status=string(status),
            obj=objective_value,
            residual_eq=residual_eq,
            residual_ineq=residual_ineq,
            residual_comp=residual_comp,
            total_time_s=solve_successful ? solve_time(jump_model) : Inf,
        ))
    end

    if save
        @info("Saving results to CSV in directory: $results_dir") 
        save_df_csv(df, results_dir, "ccopt")
    end

    return df
end
