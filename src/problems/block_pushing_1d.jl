function block_pushing_1d(;
        N=40,
        T=4.0,
        m_block=1.0,
        m_pusher=0.25,
        g=9.81,
        μ_k=0.20,
        block_half_width=0.25,
        initial_block_position=0.0,
        target_block_position=2.0,
        initial_pusher_gap=1.0,
        max_control_force=20.0,
        max_pusher_speed=3.0,
    )
    dt = T / (N - 1)

    block_position_index = 1
    block_velocity_index = 2
    pusher_position_index = 3
    pusher_velocity_index = 4
    nx, nu = 4, 1

    initial_pusher_position = initial_block_position - block_half_width - initial_pusher_gap

    x0 = [
        initial_block_position;
        0.0;
        initial_pusher_position;
        0.0;
    ]

    model = JuMP.Model()

    # x[k, :] = [block position, block velocity, pusher position, pusher velocity]
    @variable(model, x[1:N, 1:nx])

    # u[k, 1] is the force applied to the pusher point, positive right
    @variable(model, u[1:N-1])

    # contact_force[k] is the nonnegative normal force transmitted from pusher to block
    # over the interval ending at node k+1
    @variable(model, contact_force[1:N-1] >= 0)

    # Minimize the force exerted on the pusher point.
    @objective(model, Min, sum(1/2 * dt * u[k]^2 for k=1:N-1))

    @constraint(model, initial_state, x[1, :] .== x0)
    @constraint(model, terminal_block_position, x[N, block_position_index] == target_block_position)
    @constraint(model, terminal_block_velocity, x[N, block_velocity_index] == 0.0)
    @constraint(model, terminal_pusher_velocity, x[N, pusher_velocity_index] == 0.0)

    @constraint(model, control_force_lower_bound, u .>= -max_control_force)
    @constraint(model, control_force_upper_bound, u .<= max_control_force)
    @constraint(model, pusher_speed_lower_bound, x[:, pusher_velocity_index] .>= -max_pusher_speed)
    @constraint(model, pusher_speed_upper_bound, x[:, pusher_velocity_index] .<=  max_pusher_speed)

    block_position(k) = x[k, block_position_index]
    block_velocity(k) = x[k, block_velocity_index]
    pusher_position(k) = x[k, pusher_position_index]
    pusher_velocity(k) = x[k, pusher_velocity_index]

    @expression(model, block_left_side[k=1:N], block_position(k) - block_half_width)
    @expression(model, block_right_side[k=1:N], block_position(k) + block_half_width)

    @expression(model, contact_gap[k=1:N], block_left_side[k] - pusher_position(k))
    @constraint(model, initial_contact_gap_nonnegative, contact_gap[1] >= 0)
    @constraint(model, contact_gap_nonnegative[k=1:N-1], contact_gap[k+1] >= 0)

    # Get the sign of the block velocity using KKT formulation of sign fn
    @variable(model, λ_sign_kkt[k=1:N, d=1:2] >= 0)
    @variable(model, v_block_sign[k=1:N])
    
    @constraint(model, v_block_sign_kkt1[k=1:N], -block_velocity(k) + λ_sign_kkt[k, 1] - λ_sign_kkt[k, 2] == 0)
    
    # λ_sign_kkt[k, 1] ⟂ v_block_sign_kkt2_rhs[k]
    @constraint(model, v_block_sign_kkt2_rhs[k=1:N], 1 - v_block_sign[k] >= 0)
    
    # λ_sign_kkt[k, 2] ⟂ v_block_sign_kkt3_rhs[k]
    @constraint(model, v_block_sign_kkt3_rhs[k=1:N], 1 + v_block_sign[k] >= 0)
    
    # One implicit-Euler vector equality for the full coupled dynamics
    @expression(model, kinetic_friction_force[k=1:N], -v_block_sign[k] * μ_k * m_block * g)

    @expression(model, continuous_dynamics[k=1:N-1],
        [
            # d/dt block position
            block_velocity(k+1);
            # d/dt block velcity
            1/m_block * (contact_force[k] + kinetic_friction_force[k+1]);
            # d/dt pusher position
            pusher_velocity(k+1);
            # d/dt pusher velocity
            1/m_pusher * (u[k] - contact_force[k]);
        ]
    )
    @constraint(model, dynamics[k=1:N-1],
        x[k+1, :] .== x[k, :] + dt .* continuous_dynamics[k]
    )

    ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
        (:var_con, model[:contact_force], model[:contact_gap_nonnegative]),
        (:var_con, model[:λ_sign_kkt][:, 1], model[:v_block_sign_kkt2_rhs]),
        (:var_con, model[:λ_sign_kkt][:, 2], model[:v_block_sign_kkt3_rhs]),
    )

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, dt, m_block, m_pusher, g,
                       μ_k, block_half_width,
                       initial_block_position, target_block_position,
                       initial_pusher_gap, initial_pusher_position,
                       max_control_force, max_pusher_speed,
                       block_position_index, block_velocity_index,
                       pusher_position_index, pusher_velocity_index))
end

register(block_pushing_1d, :robotics)
