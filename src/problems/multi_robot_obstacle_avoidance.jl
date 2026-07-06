using Combinatorics

function multi_robot_obstacle_avoidance(;
    N=40,
    T=8.0,
    robot_clearance=1.25,
    obstacle_clearance=1.0,
    x0 = [
        0.0 1.0 0.0 0.0;
        0.0 9.0 0.0 0.0;
    ],
    xF = [
        10.0 9.0 0.0 0.0;
        10.0 1.0 0.0 0.0;
    ],
    obstacle_positions = [
        2.5 3.5;
        7.5 6.0;
    ],
    obstacle_widths = 2.0,
    obstacle_heights = 6.0,
    obstacle_angles = deg2rad.([10.0, -10.0]),
    u_max=4.0,
    v_max=4.0,
)
    Δt = T / (N - 1)
    m = 1.0

    x0 = Matrix{Float64}(x0)
    xF = Matrix{Float64}(xF)
    obstacle_positions = Matrix{Float64}(obstacle_positions)

    n_robots = size(x0, 1)
    n_obstacles = size(obstacle_positions, 1)
    @assert n_robots == 2 "multi_robot_obstacle_avoidance is set up for exactly 2 robots"
    @assert size(x0, 2) == 4 && size(xF) == size(x0) "expected x0/xF to be n_robots × 4 matrices"
    @assert size(obstacle_positions, 2) == 2 "expected obstacle_positions to be n_obstacles × 2"

    function expand_obstacle_param(v, name)
        vals = v isa Number ? fill(Float64(v), n_obstacles) : collect(Float64, v)
        length(vals) == 1 && n_obstacles > 1 && return fill(vals[1], n_obstacles)
        @assert length(vals) == n_obstacles "expected one $(name) per obstacle"
        return vals
    end

    obstacle_clearances = expand_obstacle_param(obstacle_clearance, :obstacle_clearance)
    obstacle_widths = expand_obstacle_param(obstacle_widths, :obstacle_widths)
    obstacle_heights = expand_obstacle_param(obstacle_heights, :obstacle_heights)
    obstacle_angles = expand_obstacle_param(obstacle_angles, :obstacle_angles)

    @assert all(>=(0.0), obstacle_clearances) "expected nonnegative obstacle clearances"
    @assert all(>=(0.0), obstacle_widths) "expected nonnegative obstacle widths"
    @assert all(>=(0.0), obstacle_heights) "expected nonnegative obstacle heights"
    obstacle_half_extents = hcat(0.5 .* obstacle_widths, 0.5 .* obstacle_heights)

    robot_pairs = collect(combinations(1:n_robots, 2))
    robot_obstacle_pairs = [(i, o) for i in 1:n_robots for o in 1:n_obstacles]
    obstacle_contact_obstacles = last.(robot_obstacle_pairs)
    n_robot_contacts = length(robot_pairs)
    n_obstacle_contacts = length(robot_obstacle_pairs)
    n_contacts = n_robot_contacts + n_obstacle_contacts
    clearances = [
        fill(Float64(robot_clearance), n_robot_contacts);
        [obstacle_clearances[o] for (_, o) in robot_obstacle_pairs];
    ]

    model = JuMP.Model()

    # x[k, i, :] = [px, py, vx, vy] for robot i at time node k
    @variable(model, x[1:N, 1:n_robots, 1:4])
    # u[k, i, :] = [ax, ay] control acceleration for robot i
    @variable(model, u[1:N-1, 1:n_robots, 1:2])
    # s_robot stores |relative robot position| componentwise.
    @variable(model, s_robot[1:N, 1:n_robot_contacts, 1:2])
    @variable(model, λ_robot_plus[1:N, 1:n_robot_contacts, 1:2] >= 0)
    @variable(model, λ_robot_minus[1:N, 1:n_robot_contacts, 1:2] >= 0)

    # t_obs stores the componentwise L1 distance outside each rotated rectangle:
    # t[j] = max(abs(local_position[j]) - half_extent[j], 0).
    @variable(model, t_obs[1:N, 1:n_obstacle_contacts, 1:2])
    @variable(model, λ_obs_plus[1:N, 1:n_obstacle_contacts, 1:2] >= 0)
    @variable(model, λ_obs_minus[1:N, 1:n_obstacle_contacts, 1:2] >= 0)
    @variable(model, λ_obs_zero[1:N, 1:n_obstacle_contacts, 1:2] >= 0)

    @objective(model, Min,
        sum(Δt * 0.5 * (u[k, i, :]' * u[k, i, :]) for i in 1:n_robots, k in 1:N-1)
    )

    @constraint(model, ic[i=1:n_robots, j=1:4], x[1, i, j] == x0[i, j])
    @constraint(model, fc[i=1:n_robots, j=1:4], x[N, i, j] == xF[i, j])

    continuous_dynamics(x, u; m) = [x[3:4]; u ./ m]
    @constraint(model, dynamics[k=1:N-1, i=1:n_robots],
        x[k+1, i, :] .== x[k, i, :] + Δt * continuous_dynamics(x[k+1, i, :], u[k, i, :]; m=m)
    )

    @constraint(model, u_lb[k=1:N-1, i=1:n_robots, j=1:2], u[k, i, j] >= -u_max)
    @constraint(model, u_ub[k=1:N-1, i=1:n_robots, j=1:2], u[k, i, j] <=  u_max)
    @constraint(model, v_lb[k=1:N, i=1:n_robots, j=1:2], x[k, i, 2+j] >= -v_max)
    @constraint(model, v_ub[k=1:N, i=1:n_robots, j=1:2], x[k, i, 2+j] <=  v_max)

    # Robot-robot relative position.
    @expression(model, d_robot[k=1:N, c=1:n_robot_contacts, j=1:2], begin
        p1, p2 = robot_pairs[c]
        x[k, p1, j] - x[k, p2, j]
    end)

    # Robot-obstacle relative position in each obstacle's local rotated frame.
    @expression(model, d_obs[k=1:N, c=1:n_obstacle_contacts, j=1:2], begin
        i, o = robot_obstacle_pairs[c]
        dx = x[k, i, 1] - obstacle_positions[o, 1]
        dy = x[k, i, 2] - obstacle_positions[o, 2]
        co = cos(obstacle_angles[o])
        so = sin(obstacle_angles[o])
        if j == 1
            co * dx + so * dy
        else
            -so * dx + co * dy
        end
    end)

    # Enforce s_robot = |d_robot| through KKT conditions, then require robot L1
    # separation.
    @constraint(model, kkt_robot_abs_stationarity[k=1:N, c=1:n_robot_contacts, j=1:2],
        1 - λ_robot_plus[k, c, j] - λ_robot_minus[k, c, j] == 0
    )
    @constraint(model, kkt_robot_abs_s_minus_d[k=1:N, c=1:n_robot_contacts, j=1:2],
        s_robot[k, c, j] - d_robot[k, c, j] >= 0
    )
    @constraint(model, kkt_robot_abs_s_plus_d[k=1:N, c=1:n_robot_contacts, j=1:2],
        s_robot[k, c, j] + d_robot[k, c, j] >= 0
    )
    @constraint(model, robot_l1_clearance[k=1:N, c=1:n_robot_contacts],
        sum(s_robot[k, c, j] for j=1:2) >= robot_clearance
    )

    # Enforce t_obs[j] = max(abs(d_obs[j]) - obstacle_half_extent[j], 0) through
    # KKT conditions for the epigraph of the positive part. Then require the
    # L1 distance outside the rotated rectangle to exceed obstacle_clearance.
    @constraint(model, kkt_obs_distance_stationarity[k=1:N, c=1:n_obstacle_contacts, j=1:2],
        1 - λ_obs_plus[k, c, j] - λ_obs_minus[k, c, j] - λ_obs_zero[k, c, j] == 0
    )
    @constraint(model, obs_t_minus_pos[k=1:N, c=1:n_obstacle_contacts, j=1:2],
        t_obs[k, c, j] - d_obs[k, c, j] + obstacle_half_extents[obstacle_contact_obstacles[c], j] >= 0
    )
    @constraint(model, obs_t_minus_neg[k=1:N, c=1:n_obstacle_contacts, j=1:2],
        t_obs[k, c, j] + d_obs[k, c, j] + obstacle_half_extents[obstacle_contact_obstacles[c], j] >= 0
    )
    @constraint(model, obs_t_nonnegative[k=1:N, c=1:n_obstacle_contacts, j=1:2],
        t_obs[k, c, j] >= 0
    )
    @constraint(model, obstacle_l1_clearance[k=1:N, c=1:n_obstacle_contacts],
        sum(t_obs[k, c, j] for j=1:2) >= obstacle_clearances[obstacle_contact_obstacles[c]]
    )

    mpcc = MPCC(model,
        (model[:λ_robot_plus], model[:kkt_robot_abs_s_minus_d]),
        (model[:λ_robot_minus], model[:kkt_robot_abs_s_plus_d]),
        (model[:λ_obs_plus], model[:obs_t_minus_pos]),
        (model[:λ_obs_minus], model[:obs_t_minus_neg]),
        (model[:λ_obs_zero], model[:obs_t_nonnegative]),
    )
    inds = var_inds(model)
    params = (; N, T, Δt, m, n_robots, n_obstacles, n_contacts,
                n_robot_contacts, n_obstacle_contacts,
                robot_clearance, obstacle_clearances, clearances,
                obstacle_widths, obstacle_heights, obstacle_angles, obstacle_half_extents,
                robot_pairs, robot_obstacle_pairs, obstacle_contact_obstacles,
                x0, xF, obstacle_positions, u_max, v_max)

    return (; mpcc, model, inds, params)
end

register(multi_robot_obstacle_avoidance, :robotics)
