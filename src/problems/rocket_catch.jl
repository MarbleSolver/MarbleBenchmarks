import ForwardDiff as FD

rocket_Rx(θ) = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
rocket_Ry(θ) = [cos(θ) 0 sin(θ); 0 1.0 0; -sin(θ) 0 cos(θ)]
rocket_thrust_axis_body(δr, δp) = rocket_Rx(δr) * rocket_Ry(δp) * [0.0; 0.0; 1.0]

function rocket_dynamics(x, u; m, g, l, J)
    q, v, ω = x[4:7], x[8:10], x[11:13]
    thrust_axis = rocket_thrust_axis_body(u[2], u[3])
    force_body  = u[1] * thrust_axis
    torque_body = cross([0.0; 0.0; -l], force_body)

    [
        v;
        0.5 * L(q) * _H * ω;
        [0.0; 0.0; -g] + (1 / m) * qtoQ(q) * force_body;
        J \ (torque_body - cross(ω, J * ω))
    ]
end

function linearized_rocket_dynamics(x̄, ū, Δt; m, g, l, J)
    nx, nu = 13, 3
    E = quat_state_lift(x̄[4:7], 6)
    Af = FD.jacobian(x -> rocket_dynamics(x, ū; m=m, g=g, l=l, J=J), x̄)
    Bf = FD.jacobian(u -> rocket_dynamics(x̄, u; m=m, g=g, l=l, J=J), ū)
    cf = rocket_dynamics(x̄, ū; m=m, g=g, l=l, J=J)
    sysc = ss(Af, [Bf cf], I(nx), zeros(nx, nu + 1))
    sysd = c2d(sysc, Δt, :zoh)
    return E' * sysd.A * E, E' * sysd.B[:, 1:nu], E' * sysd.B[:, end]
end

function rocket_engine_clearance(x_full, u; tower_normal)
    nozzle_axis = -qtoQ(x_full[4:7]) * rocket_thrust_axis_body(u[2], u[3])
    return tower_normal' * nozzle_axis
end

function linearized_engine_clearance(x̄, ū; tower_normal)
    E = quat_state_lift(x̄[4:7], 6)
    h0 = rocket_engine_clearance(x̄, ū; tower_normal=tower_normal)
    hx = FD.gradient(x -> rocket_engine_clearance(x, ū; tower_normal=tower_normal), x̄)
    hu = FD.gradient(u -> rocket_engine_clearance(x̄, ū + u; tower_normal=tower_normal), zeros(length(ū)))
    return h0, E' * hx, hu
end

function rocket_catch(; N=50, T=7.0)
    Δt = T / (N - 1)
    nx̃, nu = 12, 3

    m, g = 1.0, 9.81
    l = 0.7
    J = Diagonal([0.25, 0.25, 0.06])

    q0 = [1.0; 0.0; 0.0; 0.0]
    x̄ = [zeros(3); q0; zeros(6)]
    ū = [m * g; 0.0; 0.0]
    Ã, B̃, c̃ = linearized_rocket_dynamics(x̄, ū, Δt; m=m, g=g, l=l, J=J)

    catch_altitude = 2.0
    trigger_altitude = 6.0
    tower_normal = [0.0, 1.0, 0.0]
    tower_y = 0.0
    arm_half_width = 0.45
    arm_opening_angle = deg2rad(14.0)
    arm_slope = tan(arm_opening_angle)

    catch_A = [
         0.0 1.0 0.0;
         1.0 arm_slope 0.0;
        -1.0 arm_slope 0.0;
    ]
    catch_b = [tower_y; -arm_half_width; -arm_half_width]
    nh = size(catch_A, 1) + 1

    engine_h0, engine_hx, engine_hu = linearized_engine_clearance(x̄, ū; tower_normal=tower_normal)

    thrust_min, thrust_max = 0.25 * m * g, 1.8 * m * g
    gimbal_max = deg2rad(15.0)
    ϕmax = deg2rad.([25.0, 25.0, 15.0])
    ωmax = [1.2, 1.2, 0.6]

    x0_full = [
        [1.2; -5.0; 22.0];
        rptoq([0.04; -0.06; 0.0]);
        [0.0; 1.0; -1.2];
        zeros(3);
    ]
    xF_full = [
        [0.0; 1.0; catch_altitude];
        q0;
        zeros(6);
    ]
    xstart = state_to_delta(x0_full, x̄)
    xfinal = state_to_delta(xF_full, x̄)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx̃])
    @variable(model, u[1:N, 1:nu])
    @variable(model, gp[1:N] >= 0)
    @variable(model, gn[1:N] >= 0)
    @variable(model, hp[1:N, 1:nh] >= 0)
    @variable(model, hn[1:N, 1:nh] >= 0)

    stc_penalty = 10.0
    @objective(model, Min,
        sum(Δt / 2 * ((ū[1] + u[k, 1])^2 + 15.0 * (u[k, 2]^2 + u[k, 3]^2)) for k=1:N)
        + stc_penalty * Δt * sum(gp[k] + gn[k] + sum(hp[k, j] + hn[k, j] for j=1:nh) for k=1:N)
    )

    @constraint(model, ic, x[1, :] .== xstart)
    @constraint(model, tc, x[N, :] .== xfinal)
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ã * x[k, :] + B̃ * u[k, :] + c̃)

    @constraint(model, thrust_lb[k=1:N], ū[1] + u[k, 1] >= thrust_min)
    @constraint(model, thrust_ub[k=1:N], ū[1] + u[k, 1] <= thrust_max)
    @constraint(model, gimbal_lb[k=1:N, j=2:3], u[k, j] >= -gimbal_max)
    @constraint(model, gimbal_ub[k=1:N, j=2:3], u[k, j] <=  gimbal_max)
    @constraint(model, rp_lb[k=1:N, j=1:3], x[k, 3+j] >= -ϕmax[j])
    @constraint(model, rp_ub[k=1:N, j=1:3], x[k, 3+j] <=  ϕmax[j])
    @constraint(model, omega_lb[k=1:N, j=1:3], x[k, 9+j] >= -ωmax[j])
    @constraint(model, omega_ub[k=1:N, j=1:3], x[k, 9+j] <=  ωmax[j])

    @constraint(model, trigger_split[k=1:N], trigger_altitude - (x̄[3] + x[k, 3]) == gp[k] - gn[k])
    @constraint(model, catch_halfspace[k=1:N, j=1:size(catch_A, 1)],
        sum(catch_A[j, d] * (x̄[d] + x[k, d]) for d=1:3) - catch_b[j] == hp[k, j] - hn[k, j]
    )
    @expression(model, engine_clearance[k=1:N],
        engine_h0 + sum(engine_hx[d] * x[k, d] for d=1:nx̃) + sum(engine_hu[d] * u[k, d] for d=1:nu)
    )
    @constraint(model, engine_halfspace[k=1:N], engine_clearance[k] == hp[k, nh] - hn[k, nh])
    @constraint(model, stc_violation[k=1:N], sum(hn[k, j] for j=1:nh) >= 0)

    stc_pairs = var_con_complementarities(model, model[:gp], model[:stc_violation])

    ind_cc1 = first.(stc_pairs)
    ind_cc2 = last.(stc_pairs)
    cc_types = fill((:var, :con), length(stc_pairs))

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, Δt, x̄, ū, m, g, l, J, catch_altitude, trigger_altitude,
                       tower_normal, tower_y, arm_half_width, arm_opening_angle,
                       catch_A, catch_b, engine_h0, engine_hx, engine_hu,
                       thrust_min, thrust_max, gimbal_max, ϕmax, ωmax))
end

register(rocket_catch, :robotics)
