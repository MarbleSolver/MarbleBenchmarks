using MarbleBenchmarks
using Marble
using CCOpt
using MadNLP, MadNLPHSL
using NLPModels
using LinearAlgebra

include(joinpath(@__DIR__, "..", "common.jl"))

function benchmark(; problems, save=false)
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

        # TODO(@micah): Need to do this because CCOpt modifies the indices in-place, which is a bit sketchy
        ind_vcc1 = copy(problem.ind_cc1)
        ind_ccc2 = copy(problem.ind_cc2)

        mpcc = CCOpt.MPCCModelVarCon(problem.model, ind_vcc1, ind_ccc2)
        vmpcc = CCOpt.vertical_form(mpcc)

        opts = CCOpt.RelaxationOptions()
        madnlpc_solver_options = Dict(
            :bound_relax_factor=>0.0,
            :bound_push=>1e-1,
            :print_level=>MadNLP.ERROR,
            :max_iter=>4000,
            :linear_solver=>Ma27Solver,
        )
        
        solver = CCOpt.RelaxationSolver(vmpcc; solver_opts=opts, madnlpc_solver_options...)
        stats = CCOpt.solve_homotopy!(solver)

        # Compute residuals and objective value
        solve_successful = Int(stats.status) <= 2

        if solve_successful
            x = stats.solution[1:problem.model.meta.nvar] # Extract original variables, this is safe / LiftedNLPModel enforces this
        end

        objective_value = solve_successful ? Marble.obj(marble_data, x) : Inf
        residual_eq = solve_successful ? norm(Marble.residual_eq(marble_data, x), Inf) : Inf
        residual_ineq = solve_successful ? norm(Marble.residual_ineq(marble_data, x), Inf) : Inf
        residual_comp = solve_successful ? norm(Marble.residual_comp(marble_data, x), Inf) : Inf

        # Add entry to df
        push!(df, (
            problem=string(name),
            status=madnlp_status_to_string[stats.status],
            obj=objective_value,
            residual_eq=residual_eq,
            residual_ineq=residual_ineq,
            residual_comp=residual_comp,
            total_time_s=solve_successful ? stats.counters.total_time : Inf,
        ))
    end

    if save
        dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        CSV.write(joinpath(@__DIR__, "results", "ccopt_results_$dt.csv"), df)
    end
end

##

benchmark(problems=list_problems(:macmpec); save=true)
