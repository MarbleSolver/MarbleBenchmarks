function multi_robot_coordination(; N=30, T=5.0, d_min=5.0)
    Δt = T / (N - 1)
    m = 1.0
    n_robots = 3

    x0 = [Float64[-5, 0, 0, 0], Float64[0, 0, 0, 0], Float64[5, 0, 0, 0]]
    xF = [Float64[0, 10, 0, 0], Float64[5, 10, 0, 0], Float64[-5, 10, 0, 0]]

    n_slacks = binomial(n_robots, 2)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:n_robots, 1:4])
    @variable(model, u[1:N, 1:n_robots, 1:2])
    @variable(model, s[1:N, 1:n_slacks, 1:2])
    @variable(model, λ[1:N, 1:n_slacks, 1:2] >= 0)

    @objective(model, Min, sum(Δt * 0.5 * (u[k, i, :]' * u[k, i, :]) for i in 1:n_robots, k in 1:N-1))

    @constraint(model, ic[i=1:n_robots], x[1, i, :] .== x0[i])
    @constraint(model, fc[i=1:n_robots], x[N, i, :] .== xF[i])

    continuous_dynamics(x, u; m) = [x[3:4]; u ./ m]
    @constraint(model, dynamics[k=1:N-1, i=1:n_robots],
        x[k+1, i, :] .== x[k, i, :] + Δt * continuous_dynamics(x[k+1, i, :], u[k, i, :]; m=m))

    pairs = collect(combinations(1:n_robots, 2))
    @expression(model, d[k=1:N, i=1:n_slacks, j=1:2], begin
        p1, p2 = pairs[i]
        x[k, p1, j] - x[k, p2, j]
    end)

    @constraint(model, con_s_minus_d[k=1:N, i=1:n_slacks, j=1:2], s[k, i, j] - d[k, i, j] >= 0)
    @constraint(model, con_lambda_ub[k=1:N, i=1:n_slacks, j=1:2], (1 - λ[k, i, j]) >= 0)
    @constraint(model, con_s_plus_d[k=1:N, i=1:n_slacks, j=1:2], s[k, i, j] + d[k, i, j] >= 0)
    @constraint(model, [k=1:N, i=1:n_slacks], sum(s[k, i, j] for j=1:2) >= d_min)

    var_con_pairs = var_con_complementarities(model, model[:λ], model[:con_s_minus_d])
    con_con_pairs = con_con_complementarities(model, model[:con_lambda_ub], model[:con_s_plus_d])

    ind_cc1  = [first.(var_con_pairs); first.(con_con_pairs)]
    ind_cc2  = [last.(var_con_pairs);  last.(con_con_pairs)]
    cc_types = [fill((:var, :con), length(var_con_pairs)); fill((:con, :con), length(con_con_pairs))]

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, d_min, n_robots, pairs=collect(combinations(1:n_robots, 2))))
end

register(multi_robot_coordination, :robotics)
