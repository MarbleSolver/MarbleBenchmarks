# Planar 2-D hopper hopping up a short staircase.
#
# The robot is two equal point masses (foot and head) joined by a prismatic leg
# actuated by an internal force. The configuration stacks the [x; z] position of
# each mass (foot, then head), and the state appends their velocities, giving
#
#   q = [foot_x, foot_z, head_x, head_z]        (nq = 4)
#   x = [q; q̇]                                  (nx = 8)
#   u = [u_x, u_z]                              (nu = 2, internal leg force)
#
# Contact with the terrain is modeled with complementarity: a nonnegative normal
# force is complementary to the foot's signed distance plus slip, and the terrain
# height under the foot is selected by four box-constrained "step" variables whose
# activation is itself expressed through complementarity. All of these are con-con
# complementarities registered on the MPCC.
function hopper_2d(;
        N=60,
        dt=0.05,
        μ=0.1,
        speed=0.047,
        g=9.81,
        u_scale=90.0,
        f_scale=30.0,
        s_fric_scale=20.0,
    )
    nq, nx, nu, nd = 4, 8, 2, 2       # nd = number of spatial dimensions (2 = planar)
    pos_xi, pos_zi = 1, 2             # indexing into a spatial coordinate

    grav_comp = [0.0; g]             # gravity compensation added to the leg force

    # Cost weights: track the foot/head x-positions and hold the head height at the
    # terminal node, with light regularization on velocities and controls.
    Q  = diagm([1e1; 0; 1e1; 0; 1e-1 * ones(4)])
    Qf = diagm([1e3; 0; 1e3; 1e3; 1e-2 * ones(4)])
    R  = diagm([1e-3; 1e-3]) / u_scale / u_scale
    Rd = diagm([1e0; 1e0]) / u_scale

    # Reference: walk forward at `speed` per step and finish a bit further ahead.
    x_goal = [[[speed * k; 0.0; speed * k; 0.0; zeros(4)] for k in 0:N-1]...,
              [speed * N + 0.3; 0.0; speed * N + 0.3; 0.0; zeros(4)]]

    # Staircase geometry: three rising steps and a final wall.
    step1_start, step_height1 = 0.5, 0.1
    step_start2, step_height2 = 0.75, 0.1
    step_start3, step_height3 = 1.0, 0.1
    step_end, step_end_height = 2.0, 0.3

    model = JuMP.Model()

    # x[k, :] = [foot_x, foot_z, head_x, head_z, foot_ẋ, foot_ż, head_ẋ, head_ż]
    @variable(model, x[1:N+1, 1:nx])
    # u[k, :] internal leg force (scaled by u_scale)
    @variable(model, u[1:N, 1:nu])
    # f[k, :] contact force [tangential; normal] at the foot (scaled by f_scale)
    @variable(model, f[1:N, 1:nd])
    # s_fric[k, :] nonnegative +x / -x tangential slip of the foot
    @variable(model, s_fric[1:N, 1:2])
    # s_step[k, :] four (height, upper-slack, lower-slack) triples selecting the
    # active step under the foot; columns are [h1, h1_l1, h1_l2, ..., h4, h4_l1, h4_l2]
    @variable(model, s_step[1:N, 1:12])

    q_foot, q_head = x[:, 1:nd], x[:, nd .+ (1:nd)]
    v_foot, v_head = x[:, 2*nd .+ (1:nd)], x[:, 3*nd .+ (1:nd)]

    @objective(model, Min,
        # tracking + control effort
        0.5 * sum((x[k, :] - x_goal[k])' * Q * (x[k, :] - x_goal[k]) + u[k, :]' * R * u[k, :] for k = 1:N) +
        # terminal cost
        0.5 * (x[end, :] - x_goal[end])' * Qf * (x[end, :] - x_goal[end]) +
        # keep the prismatic leg short
        0.5 * sum((q_foot[k+1, pos_zi] - q_head[k+1, pos_zi])^2 * 1e1 for k = 1:N) +
        # control smoothing
        0.5 * sum((u[k+1, :] - u[k, :])' * Rd * (u[k+1, :] - u[k, :]) * 10 for k in 1:N-1) +
        # penalize slip
        0.5 * sum((s_fric[k, :])' * s_fric[k, :] / s_fric_scale / s_fric_scale for k in 1:N)
    )

    @constraint(model, initial_state, x[1, :] .== zeros(nx))

    # Rescale slip for conditioning; downstream constraints use the scaled version.
    s_fric = s_fric ./ s_fric_scale

    # Backwards-Euler dynamics. The head carries the leg force + gravity comp, the
    # foot carries the reaction plus the contact force.
    @constraint(model, foot_position[k=1:N], q_foot[k+1, :] .== q_foot[k, :] + v_foot[k+1, :] * dt)
    @constraint(model, head_position[k=1:N], q_head[k+1, :] .== q_head[k, :] + v_head[k+1, :] * dt)
    @constraint(model, foot_velocity[k=1:N],
        v_foot[k+1, :] / dt .== v_foot[k, :] / dt + (-[0; g] + f[k, :] * f_scale - (u[k, :] * u_scale + grav_comp)))
    @constraint(model, head_velocity[k=1:N],
        v_head[k+1, :] / dt .== v_head[k, :] / dt + (-[0; g] + (u[k, :] * u_scale + grav_comp)))

    # s_step columns: (height activation, upper-bound slack, lower-bound slack) per step.
    h1, h1_l1, h1_l2, h2, h2_l1, h2_l2, h3, h3_l1, h3_l2, h4, h4_l1, h4_l2 = [s_step[:, i] for i in 1:12]

    # Stationarity relating the foot x-position to each step's edge, split into the
    # nonnegative slacks of the surrounding box constraint.
    @constraint(model, step1_stationarity[k=1:N], q_foot[k+1, pos_xi] - step1_start .== h1_l1[k] - h1_l2[k])
    @constraint(model, step2_stationarity[k=1:N], q_foot[k+1, pos_xi] - step_start2 .== h2_l1[k] - h2_l2[k])
    @constraint(model, step3_stationarity[k=1:N], q_foot[k+1, pos_xi] - step_start3 .== h3_l1[k] - h3_l2[k])
    @constraint(model, step4_stationarity[k=1:N], q_foot[k+1, pos_xi] - step_end   .== h4_l1[k] - h4_l2[k])

    # Box constraints -1 <= h_i <= 1 whose active side is picked via complementarity
    # with the slacks above (upper: h_l1 ⟂ 1 - h_i, lower: h_l2 ⟂ h_i + 1).
    @constraint(model, step1_ub_comp1[k=1:N], h1_l1[k] >= 0)
    @constraint(model, step1_ub_comp2[k=1:N], 1 >= h1[k])
    @constraint(model, step1_lb_comp1[k=1:N], h1_l2[k] >= 0)
    @constraint(model, step1_lb_comp2[k=1:N], h1[k] >= -1)

    @constraint(model, step2_ub_comp1[k=1:N], h2_l1[k] >= 0)
    @constraint(model, step2_ub_comp2[k=1:N], 1 >= h2[k])
    @constraint(model, step2_lb_comp1[k=1:N], h2_l2[k] >= 0)
    @constraint(model, step2_lb_comp2[k=1:N], h2[k] >= -1)

    @constraint(model, step3_ub_comp1[k=1:N], h3_l1[k] >= 0)
    @constraint(model, step3_ub_comp2[k=1:N], 1 >= h3[k])
    @constraint(model, step3_lb_comp1[k=1:N], h3_l2[k] >= 0)
    @constraint(model, step3_lb_comp2[k=1:N], h3[k] >= -1)

    @constraint(model, step4_ub_comp1[k=1:N], h4_l1[k] >= 0)
    @constraint(model, step4_ub_comp2[k=1:N], 1 >= h4[k])
    @constraint(model, step4_lb_comp1[k=1:N], h4_l2[k] >= 0)
    @constraint(model, step4_lb_comp2[k=1:N], h4[k] >= -1)

    # Signed distance from the foot to the terrain: the base floor minus the active
    # step heights plus the final wall.
    @expression(model, height[k=1:N],
        q_foot[k+1, pos_zi]
        - 0.5 * (h1[k] + 1) * step_height1
        - 0.5 * (h2[k] + 1) * step_height2
        - 0.5 * (h3[k] + 1) * step_height3
        + 0.5 * (h4[k] + 1) * step_end_height)
    @constraint(model, foot_above_terrain[k=1:N], height[k] >= 0)

    # Coulomb friction: horizontal foot motion is the difference of the +x / -x slips,
    # and the tangential force lies inside the friction cone.
    @constraint(model, friction_motion[k=1:N], q_foot[k+1, pos_xi] - q_foot[k, pos_xi] .== s_fric[k, 1] - s_fric[k, 2])
    @constraint(model, friction_cone_ub[k=1:N], μ * f[k, pos_zi] .>= f[k, pos_xi])
    @constraint(model, friction_cone_lb[k=1:N], f[k, pos_xi] .>= -μ * f[k, pos_zi])
    @constraint(model, slip_nonnegative, s_fric .>= 0)

    # Contact complementarity: a nonnegative normal force is complementary to the
    # foot's signed distance plus the slip magnitude.
    @constraint(model, contact_normal_f_comp[k=1:N], f[k, pos_zi] >= 0)
    @constraint(model, contact_normal_x_comp[k=1:N], height[k] + s_fric[k, 1] + s_fric[k, 2] >= 0)

    mpcc = MPCC(model,
        # normal force ⟂ signed distance + slip
        (model[:contact_normal_x_comp], model[:contact_normal_f_comp]),
        # step activation box complementarities (upper/lower per step)
        (model[:step1_ub_comp1], model[:step1_ub_comp2]),
        (model[:step1_lb_comp1], model[:step1_lb_comp2]),
        (model[:step2_ub_comp1], model[:step2_ub_comp2]),
        (model[:step2_lb_comp1], model[:step2_lb_comp2]),
        (model[:step3_ub_comp1], model[:step3_ub_comp2]),
        (model[:step3_lb_comp1], model[:step3_lb_comp2]),
        (model[:step4_ub_comp1], model[:step4_ub_comp2]),
        (model[:step4_lb_comp1], model[:step4_lb_comp2]),
    )
    inds = var_inds(model)
    params = (; N, dt, μ, speed, g, u_scale, f_scale, s_fric_scale,
                nq, nx, nu, nd, pos_xi, pos_zi,
                step1_start, step_height1, step_start2, step_height2,
                step_start3, step_height3, step_end, step_end_height,
                x_goal)

    return (; mpcc, model, inds, params)
end

register(hopper_2d, :robotics)
