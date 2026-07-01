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
        p = get_problem(name)
        if !(p isa Marble.MPCC) 
            @assert p isa NamedTuple "get_problem($name) must return a NamedTuple or a Marble.MPCC"
            v = values(p)
            mpcc = v[findfirst(x -> x isa Marble.MPCC, v)]
        else
            mpcc = p
        end

        data = Marble.mpcc_to_marble(mpcc)
        problem = Marble.Problem(data...)

        solver = Marble.Solver()
        Marble.setup!(solver, mpcc;
                      verbosity=0,
                      max_iters=2000,
                      relax_min=1e-8,
                      penalty_max=1e9,
                      relax_scaling=0.1,
                      use_relax_correction=true,
                      convergence_eq_violation=5e-8,
                      convergence_ineq_violation=5e-8,
                      convergence_comp_violation=5e-8
        )
        result = Marble.solve!(solver)

        solve_successful = Marble.converged(result)
        x = solve_successful ? Marble.z(result) : nothing

        stats = solution_stats(problem, x; solve_successful=solve_successful)
        status = solve_successful ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE
        
        # Add entry to df
        push!(df, (
            problem=string(name),
            status=string(status),
            obj=stats.objective_value,
            residual_eq=stats.residual_eq,
            residual_ineq=stats.residual_ineq,
            residual_comp=stats.residual_comp,
            total_time_s=solve_successful ? Marble.solve_time_s(result) : Inf,
        ))
    end

    if save
        @info("Saving results to CSV in directory: $results_dir") 
        save_df_csv(df, results_dir, "marble")
    end
    
    return df
end