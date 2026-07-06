import ForwardDiff as FD

rocket_Rx(θ) = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
rocket_Ry(θ) = [cos(θ) 0 sin(θ); 0 1.0 0; -sin(θ) 0 cos(θ)]
rocket_thrust_axis_body(δr, δp) = rocket_Rx(δr) * rocket_Ry(δp) * [0.0; 0.0; 1.0]

# Full state: [p; q; v; ω; δr; δp]. Control: [thrust; δ̇r; δ̇p]
function rocket_dynamics(x, u; m, g, l, J)
    q, v, ω = x[4:7], x[8:10], x[11:13]
    δr, δp = x[14], x[15]
    thrust_axis = rocket_thrust_axis_body(δr, δp)
    force_body  = u[1] * thrust_axis
    torque_body = cross([0.0; 0.0; -l], force_body)

    [
        v;
        0.5 * L(q) * _H * ω;
        [0.0; 0.0; -g] + (1 / m) * qtoQ(q) * force_body;
        J \ (torque_body - cross(ω, J * ω));
        u[2];
        u[3];
    ]
end

function linearized_rocket_dynamics(x̄, ū, Δt; m, g, l, J)
    nx, nu = 15, 3
    E = quat_state_lift(x̄[4:7], 8)
    Af = FD.jacobian(x -> rocket_dynamics(x, ū; m=m, g=g, l=l, J=J), x̄)
    Bf = FD.jacobian(u -> rocket_dynamics(x̄, u; m=m, g=g, l=l, J=J), ū)
    cf = rocket_dynamics(x̄, ū; m=m, g=g, l=l, J=J)
    sysc = ss(Af, [Bf cf], I(nx), zeros(nx, nu + 1))
    sysd = c2d(sysc, Δt, :zoh)
    return E' * sysd.A * E, E' * sysd.B[:, 1:nu], E' * sysd.B[:, end]
end

function first_order_plume_axis_world(ϕ, δr, δp)
    [
        -δp - 2 * ϕ[2];
         δr + 2 * ϕ[1];
        -1.0;
    ]
end

function first_order_plume_clearance(ϕ, δr, δp; tower_normal)
    return tower_normal' * first_order_plume_axis_world(ϕ, δr, δp)
end

function rocket_catch_3d(; N=40, T=16.0,
    # Units: km, s, tonne, MN. Defaults are scaled Super Heavy-style
    x0 = [0.080, 0.120, 0.550],
    q0 = rptoq(deg2rad.([1.5, -2.0, 0.0])),
    v0 = [-0.004, -0.012, -0.085],
    w0 = zeros(3),
    catch_altitude = 0.080,
    trigger_altitude = 0.120,
    terminal_y_min = 0.012,
    terminal_y_max = 0.030,
    terminal_y_weight = 1.0e5,
)
    Δt = T / (N - 1)
    nx̃, nu = 14, 3

    # Scaled Super Heavy-ish landing parameters
    m, g = 300.0, 9.81e-3

    booster_height = 0.071        # 71 m
    booster_radius = 0.0045       # 9 m diameter
    l = 0.035                     # CoM to engine plane, about half the booster

    # Slender-cylinder inertia
    J_roll_pitch = (1 / 12) * m * (3 * booster_radius^2 + booster_height^2)
    J_yaw = 0.5 * m * booster_radius^2
    J = Diagonal([J_roll_pitch, J_roll_pitch, J_yaw])

    q_ref = [1.0; 0.0; 0.0; 0.0]
    # Reduced state: [p; ϕ; v; ω; δr; δp]
    x̄ = [zeros(3); q_ref; zeros(8)]
    ū = [m * g; 0.0; 0.0]
    Ã, B̃, c̃ = linearized_rocket_dynamics(x̄, ū, Δt; m=m, g=g, l=l, J=J)

    # Catch funnel and trigger geometry
    tower_normal = [0.0, 1.0, 0.0]
    tower_y = 0.0
    arm_half_width = 0.006        # 12 m gap at the tower face
    arm_opening_angle = deg2rad(12.0)
    arm_slope = tan(arm_opening_angle)

    # Funnel halfspaces: y >= tower_y and |x| <= arm_half_width + arm_slope*y
    catch_A = [
         0.0 1.0 0.0;
         1.0 arm_slope 0.0;
        -1.0 arm_slope 0.0;
    ]
    catch_b = [tower_y; -arm_half_width; -arm_half_width]
    n_catch_halfspaces = size(catch_A, 1)
    plume_away_from_tower_index = n_catch_halfspaces + 1
    n_state_triggered_constraints = plume_away_from_tower_index
    nh = n_state_triggered_constraints

    raptor2_sea_level_thrust = 2.26
    landing_engines_max = 13
    thrust_min = 0.25 * m * g
    thrust_max = landing_engines_max * raptor2_sea_level_thrust
    gimbal_max = deg2rad(15.0)
    gimbal_rate_max = deg2rad(20.0)
    ϕ_max = deg2rad.([18.0, 18.0, 10.0])
    ω_max = deg2rad.([12.0, 12.0, 12.0])

    @assert length(x0) == 3 && length(q0) == 4 && length(v0) == 3 && length(w0) == 3 "expected x0(3), q0(4), v0(3), w0(3)"
    @assert terminal_y_min <= terminal_y_max "expected terminal_y_min <= terminal_y_max"
    # Start with centered gimbal angles
    x0_full = [x0; q0; v0; w0; 0.0; 0.0]
    xF_full = [
        [0.0; terminal_y_min; catch_altitude];
        q_ref;
        zeros(6);
        [0.0; 0.0];   # gimbal δr, δp centered at the catch
    ]
    xstart = state_to_delta(x0_full, x̄)
    xfinal = state_to_delta(xF_full, x̄)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx̃])
    @variable(model, u[1:N-1, 1:nu])
    # Trigger and margin slacks
    @variable(model, gp[1:N] >= 0)
    @variable(model, gn[1:N] >= 0)
    @variable(model, hp[1:N, 1:nh] >= 0)
    @variable(model, hn[1:N, 1:nh] >= 0)

    stc_penalty = 1.0
    gimbal_rate_weight = 1.0
    gimbal_angle_weight = 15.0
    @objective(model, Min,
        sum(Δt / 2 * ((ū[1] + u[k, 1])^2 + gimbal_rate_weight * (u[k, 2]^2 + u[k, 3]^2)) for k=1:N-1)
        + sum(Δt / 2 * gimbal_angle_weight * (x[k, 13]^2 + x[k, 14]^2) for k=1:N)
        + terminal_y_weight / 2 * (x[N, 2] - terminal_y_min)^2
        + stc_penalty * Δt * sum(gp[k] + gn[k] + sum(hp[k, j] + hn[k, j] for j=1:nh) for k=1:N)
    )

    @constraint(model, ic, x[1, :] .== xstart)
    # Terminal y can land anywhere in the catch corridor
    terminal_exact_inds = [1; collect(3:nx̃)]
    @constraint(model, tc[j=terminal_exact_inds], x[N, j] == xfinal[j])
    @constraint(model, tc_y_lb, x[N, 2] >= terminal_y_min)
    @constraint(model, tc_y_ub, x[N, 2] <= terminal_y_max)
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ã * x[k, :] + B̃ * u[k, :] + c̃)

    @constraint(model, thrust_lb[k=1:N-1], ū[1] + u[k, 1] >= thrust_min)
    @constraint(model, thrust_ub[k=1:N-1], ū[1] + u[k, 1] <= thrust_max)
    # Gimbal rate, gimbal angle, attitude, and angular-rate bounds
    @constraint(model, gimbal_rate_lb[k=1:N-1, j=2:3], u[k, j] >= -gimbal_rate_max)
    @constraint(model, gimbal_rate_ub[k=1:N-1, j=2:3], u[k, j] <=  gimbal_rate_max)
    @constraint(model, gimbal_lb[k=1:N, j=13:14], x[k, j] >= -gimbal_max)
    @constraint(model, gimbal_ub[k=1:N, j=13:14], x[k, j] <=  gimbal_max)
    @constraint(model, rp_lb[k=1:N, j=1:3], x[k, 3+j] >= -ϕ_max[j])
    @constraint(model, rp_ub[k=1:N, j=1:3], x[k, 3+j] <=  ϕ_max[j])
    @constraint(model, ω_lb[k=1:N, j=1:3], x[k, 9+j] >= -ω_max[j])
    @constraint(model, ω_ub[k=1:N, j=1:3], x[k, 9+j] <=  ω_max[j])

    # Triggered catch constraints
    @expression(model, rocket_CoM[k=1:N, d=1:3], x̄[d] + x[k, d])

    # Split trigger_altitude - z into gp - gn
    @constraint(model, trigger_altitude_split[k=1:N],
        trigger_altitude - rocket_CoM[k, 3] == gp[k] - gn[k]
    )

    # Funnel margins
    @expression(model, catch_halfspace_margin[k=1:N, j=1:n_catch_halfspaces],
        sum(catch_A[j, d] * rocket_CoM[k, d] for d=1:3) - catch_b[j]
    )
    @constraint(model, catch_halfspace[k=1:N, j=1:n_catch_halfspaces],
        catch_halfspace_margin[k, j] == hp[k, j] - hn[k, j]
    )

    # First-order plume margin
    @expression(model, plume_axis_world[k=1:N], first_order_plume_axis_world(x[k, 4:6], x[k, 13], x[k, 14]))
    @expression(model, plume_away_from_tower_margin[k=1:N], tower_normal' * plume_axis_world[k])
    @constraint(model, engine_plume_away_from_tower[k=1:N],
        plume_away_from_tower_margin[k] ==
        hp[k, plume_away_from_tower_index] -
        hn[k, plume_away_from_tower_index]
    )

    # Aggregate triggered violations
    @constraint(model, total_hn[k=1:N],
        sum(hn[k, j] for j=1:n_state_triggered_constraints) >= 0
    )

    # Complementarity pair: below trigger vs. constraint violation
    mpcc = MPCC(model, (model[:gp], model[:total_hn]))
    inds = var_inds(model)
    params = (; N, T, Δt, x̄, ū, m, g, l, J, booster_height, booster_radius,
                raptor2_sea_level_thrust, landing_engines_max,
                catch_altitude, trigger_altitude,
                terminal_y_min, terminal_y_max, terminal_y_weight,
                tower_normal, tower_y, arm_half_width, arm_opening_angle,
                catch_A, catch_b,
                thrust_min, thrust_max, gimbal_max, gimbal_rate_max, ϕ_max, ω_max)
    
    return (; mpcc, model, inds, params)
end

register(rocket_catch_3d, :robotics)
