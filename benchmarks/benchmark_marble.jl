function benchmark_marble(; problems, save=false, results_dir=joinpath(@__DIR__, "results"))
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

        nlp = problem.model isa JuMP.Model ? MathOptNLPModel(problem.model) : problem.model
        marble_data = MarbleData(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

        # Set tolerance options
        opts = Marble.SolverOptions()

        Marble.verbosity!(opts, 0)
        Marble.penalty_max!(opts, 1e7)
        Marble.relaxation_min!(opts, 1e-9)
        Marble.convergence_eq_violation!(opts, 1e-8)
        Marble.convergence_ineq_violation!(opts, 1e-8)
        Marble.convergence_comp_violation!(opts, 1e-8)

        result = Marble.solve(marble_data; opts = opts)
        status = Marble.converged(result) ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE

        # Compute residuals and objective value
        solve_successful = status in (MOI.LOCALLY_SOLVED, MOI.LOCALLY_SOLVED)

        if solve_successful
            x = Marble.z(result)
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
            total_time_s=solve_successful ? Marble.solve_time_s(result) : Inf,
        ))
    end

    if save
        @info("Saving results to CSV in directory: $results_dir") 
        save_df_csv(df, results_dir, "ccopt")
    end
    
    return df
end