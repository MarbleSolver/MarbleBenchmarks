function marble_solve(name, options=(;))
    mpcc = get_mpcc(name)
    data = Marble.mpcc_to_marble(mpcc)
    solver = Marble.Solver()
    Marble.setup!(solver, mpcc; options...)

    x, converged, solve_time = try
        result = Marble.solve!(solver)
        converged = Marble.converged(result)
        # `Marble.z` returns a Julia array that ALIASES the C++-owned solution buffer
        # inside `result`/`solver`. Once those go out of scope and are GC'd the buffer
        # is freed (and later reused by the next solve), so the escaping vector must be
        # copied into Julia-owned memory or it decays into garbage before serialization
        x = copy(Marble.z(result))
        solve_time = Marble.solve_time_s(result)
        (x, converged, solve_time)
    catch e
        (nothing, false, Inf)
    end

    feas = residuals_or_failed(data, x)
    BenchmarkProblemResult(;
        converged,
        status = string(converged ? MOI.LOCALLY_SOLVED : MOI.INFEASIBLE),
        solve_time = solve_time,
        x,
        feas...,
    )
end

