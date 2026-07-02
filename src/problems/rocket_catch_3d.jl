import ForwardDiff as FD

rocket_Rx(θ) = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
rocket_Ry(θ) = [cos(θ) 0 sin(θ); 0 1.0 0; -sin(θ) 0 cos(θ)]
rocket_thrust_axis_body(δr, δp) = rocket_Rx(δr) * rocket_Ry(δp) * [0.0; 0.0; 1.0]

# State: x = [p(3); q(4); v(3); ω(3); δr; δp] (15). The gimbal roll/pitch
# angles (δr, δp) are now states. Controls: u = [thrust; δṙ; δṗ] (3), so the
# gimbal angles integrate the commanded gimbal rates.
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
        u[2];   # δ̇r — gimbal roll rate
        u[3];   # δ̇p — gimbal pitch rate
    ]
end

function linearized_rocket_dynamics(x̄, ū, Δt; m, g, l, J)
    nx, nu = 15, 3
    E = quat_state_lift(x̄[4:7], 8)   # tail = v(3) + ω(3) + gimbal δr, δp
    Af = FD.jacobian(x -> rocket_dynamics(x, ū; m=m, g=g, l=l, J=J), x̄)
    Bf = FD.jacobian(u -> rocket_dynamics(x̄, u; m=m, g=g, l=l, J=J), ū)
    cf = rocket_dynamics(x̄, ū; m=m, g=g, l=l, J=J)
    sysc = ss(Af, [Bf cf], I(nx), zeros(nx, nu + 1))
    sysd = c2d(sysc, Δt, :zoh)
    return E' * sysd.A * E, E' * sysd.B[:, 1:nu], E' * sysd.B[:, end]
end

"""
    first_order_plume_axis_world(ϕ, δr, δp)

First-order (small-angle) world-frame engine-plume axis, used by the plume
state-triggered constraint below.

Derivation:
  • Body-frame thrust axis is `Rx(δr) * Ry(δp) * e₃`. Linearizing the gimbal
    rotations in the roll/pitch angles `(δr, δp)` gives
        thrust_body ≈ e₃ + [δp, -δr, 0].
  • The plume exhausts opposite to the thrust, so
        plume_body ≈ -(e₃ + [δp, -δr, 0]) = [-δp, δr, -1].
  • Attitude is stored as the small quaternion/Rodrigues vector `ϕ = qᵥ / qₛ`,
    whose equivalent rotation vector is `2ϕ`, so the body→world map linearizes
    to `R ≈ I + 2[ϕ]×`. Applying it and dropping second-order (ϕ·δ) products:
        plume_world ≈ plume_body + 2ϕ × (-e₃)
                    = [-δp, δr, -1] + [-2ϕ₂, 2ϕ₁, 0]
                    = [-δp - 2ϕ₂, δr + 2ϕ₁, -1].
"""
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

# Initial condition can be set per-component: position `x0`, attitude quaternion
# `q0`, linear velocity `v0`, and angular velocity `w0`. The initial gimbal is
# always centered (δr = δp = 0). Defaults reproduce the nominal descent start.
function rocket_catch_3d(; N=40, T=4.0,
    x0 = [5.0, 5.0, 22.0],
    q0 = rptoq([0.04, -0.06, 0.0]),
    v0 = [0.0, 1.0, -1.2],
    w0 = zeros(3),
)
    Δt = T / (N - 1)
    nx̃, nu = 14, 3

    m, g = 1.0, 9.81
    l = 0.7
    J = Diagonal([0.25, 0.25, 0.06])

    q_ref = [1.0; 0.0; 0.0; 0.0]
    # Reference/trim state: hover, upright, gimbal centered. Full state tail is
    # v(3), ω(3), δr, δp — all zero at trim. Reduced (delta) layout is
    # [p(1:3); ϕ(4:6); v(7:9); ω(10:12); δr(13); δp(14)].
    x̄ = [zeros(3); q_ref; zeros(8)]
    ū = [m * g; 0.0; 0.0]   # trim controls: hover thrust, zero gimbal rates
    Ã, B̃, c̃ = linearized_rocket_dynamics(x̄, ū, Δt; m=m, g=g, l=l, J=J)

    # Catch-tower geometry. The tower stands in the plane y = tower_y with its
    # face normal pointing toward the incoming rocket (tower_normal = +y). The
    # catch arms form a horizontal funnel in the x–y plane: half-width
    # arm_half_width at the tower face, opening outward (toward +y, away from the
    # tower) at half-angle arm_opening_angle.
    catch_altitude = 2.0        # z the CoM should be caught at
    trigger_altitude = 6.0      # z below which the catch constraints switch on
    tower_normal = [0.0, 1.0, 0.0]
    tower_y = 0.0
    arm_half_width = 0.45
    arm_opening_angle = deg2rad(14.0)
    arm_slope = tan(arm_opening_angle)

    # Rows encode `catch_A[j, :] ⋅ CoM ≥ catch_b[j]`, i.e. the funnel:
    #   1) y ≥ tower_y                       -> stay on the rocket's side of the
    #                                           tower face (never cross into it)
    #   2) x + arm_slope·y ≥ -arm_half_width -> right of the left funnel wall
    #   3) -x + arm_slope·y ≥ -arm_half_width-> left of the right funnel wall
    # Rows 2–3 together give |x| ≤ arm_half_width + arm_slope·y (a wedge that
    # widens with distance y from the tower face). Altitude z is unconstrained
    # here; it is handled by the trigger.
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

    thrust_min, thrust_max = 0.25 * m * g, 1.8 * m * g
    gimbal_max = deg2rad(15.0)         # limit on the gimbal angle STATE
    gimbal_rate_max = deg2rad(60.0)    # limit on the gimbal rate CONTROL
    ϕ_max = deg2rad.([25.0, 25.0, 15.0])
    ω_max = [1.2, 1.2, 0.6]

    @assert length(x0) == 3 && length(q0) == 4 && length(v0) == 3 && length(w0) == 3 "expected x0(3), q0(4), v0(3), w0(3)"
    # Assemble the full initial state; the gimbal always starts centered (0, 0).
    x0_full = [x0; q0; v0; w0; 0.0; 0.0]
    xF_full = [
        [0.0; 1.0; catch_altitude];
        q_ref;
        zeros(6);
        [0.0; 0.0];   # gimbal δr, δp centered at the catch
    ]
    xstart = state_to_delta(x0_full, x̄)
    xfinal = state_to_delta(xF_full, x̄)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx̃])
    @variable(model, u[1:N-1, 1:nu])
    # STC slacks. gp/gn split the trigger signal into its positive/negative
    # parts (gp > 0 ⇔ rocket is below trigger_altitude). hp/hn split each
    # triggered-constraint margin into satisfied part (hp) and violation (hn).
    @variable(model, gp[1:N] >= 0)
    @variable(model, gn[1:N] >= 0)
    @variable(model, hp[1:N, 1:nh] >= 0)
    @variable(model, hn[1:N, 1:nh] >= 0)

    stc_penalty = 1.0
    gimbal_rate_weight = 1.0    # control effort on the gimbal rates u[k, 2:3]
    gimbal_angle_weight = 15.0  # keep the gimbal-angle states x[k, 13:14] small
    @objective(model, Min,
        sum(Δt / 2 * ((ū[1] + u[k, 1])^2 + gimbal_rate_weight * (u[k, 2]^2 + u[k, 3]^2)) for k=1:N-1)
        + sum(Δt / 2 * gimbal_angle_weight * (x[k, 13]^2 + x[k, 14]^2) for k=1:N)
        + stc_penalty * Δt * sum(gp[k] + gn[k] + sum(hp[k, j] + hn[k, j] for j=1:nh) for k=1:N)
    )

    @constraint(model, ic, x[1, :] .== xstart)
    @constraint(model, tc, x[N, :] .== xfinal)
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ã * x[k, :] + B̃ * u[k, :] + c̃)

    @constraint(model, thrust_lb[k=1:N-1], ū[1] + u[k, 1] >= thrust_min)
    @constraint(model, thrust_ub[k=1:N-1], ū[1] + u[k, 1] <= thrust_max)
    # Gimbal RATE limits on the controls u[k, 2:3]...
    @constraint(model, gimbal_rate_lb[k=1:N-1, j=2:3], u[k, j] >= -gimbal_rate_max)
    @constraint(model, gimbal_rate_ub[k=1:N-1, j=2:3], u[k, j] <=  gimbal_rate_max)
    # ...and gimbal ANGLE limits on the states x[k, 13:14].
    @constraint(model, gimbal_lb[k=1:N, j=13:14], x[k, j] >= -gimbal_max)
    @constraint(model, gimbal_ub[k=1:N, j=13:14], x[k, j] <=  gimbal_max)
    @constraint(model, rp_lb[k=1:N, j=1:3], x[k, 3+j] >= -ϕ_max[j])
    @constraint(model, rp_ub[k=1:N, j=1:3], x[k, 3+j] <=  ϕ_max[j])
    @constraint(model, ω_lb[k=1:N, j=1:3], x[k, 9+j] >= -ω_max[j])
    @constraint(model, ω_ub[k=1:N, j=1:3], x[k, 9+j] <=  ω_max[j])

    # ---------------------------------------------------------------------
    # State-triggered constraints (STCs), encoded as an MPCC.
    #
    # The catch constraints should be enforced only once the rocket has
    # descended near the tower. We express that with a single complementarity
    # pair `gp ⟂ total_hn` (see the MPCC(...) call at the end):
    #
    #   • Trigger: gp[k] is the positive part of (trigger_altitude - CoM_z),
    #     so gp[k] > 0 exactly when the rocket is BELOW trigger_altitude.
    #   • Constraints: hn[k, j] is the violation (negative-margin) slack of the
    #     j-th triggered constraint, and total_hn[k] = Σⱼ hn[k, j].
    #   • Complementarity gp[k] · total_hn[k] = 0 then means: whenever the rocket
    #     is below the trigger (gp > 0), every violation slack must be zero
    #     (total_hn = 0 ⇒ all hn = 0 ⇒ all margins ≥ 0). Above the trigger
    #     (gp = 0) the margins are free and the catch geometry is ignored.
    # ---------------------------------------------------------------------
    @expression(model, rocket_CoM[k=1:N, d=1:3], x̄[d] + x[k, d])

    # STC #0 — the trigger. Exact positive/negative split of the altitude gap:
    #   trigger_altitude - CoM_z = gp - gn,  gp, gn ≥ 0.
    # gp is the "below trigger" amount and feeds the complementarity above; gn
    # is the "above trigger" amount and is otherwise unused.
    @constraint(model, trigger_altitude_split[k=1:N],
        trigger_altitude - rocket_CoM[k, 3] == gp[k] - gn[k]
    )

    # STCs #1..3 — when active, the rocket CoM must lie inside the catch funnel:
    #   catch_A[j, :] ⋅ CoM ≥ catch_b[j]   for every catch half-space j.
    # Writing each signed margin as hp - hn (both ≥ 0) lets the complementarity
    # drive hn → 0 (margin ≥ 0) only while the trigger is active.
    @expression(model, catch_halfspace_margin[k=1:N, j=1:n_catch_halfspaces],
        sum(catch_A[j, d] * rocket_CoM[k, d] for d=1:3) - catch_b[j]
    )
    @constraint(model, catch_halfspace[k=1:N, j=1:n_catch_halfspaces],
        catch_halfspace_margin[k, j] == hp[k, j] - hn[k, j]
    )

    # STC #4 — when active, the engine plume must point away from the tower face.
    # first_order_plume_axis_world gives the small-angle world plume direction
    # from the attitude ϕ = x[k, 4:6] and the gimbal-angle states δr = x[k, 13],
    # δp = x[k, 14]. Projecting onto tower_normal (= +y) yields the clearance
    # margin
    #   plume · tower_normal = δr + 2ϕ₁,
    # which the STC keeps ≥ 0 so the exhaust never blows back onto the tower.
    @expression(model, plume_axis_world[k=1:N], first_order_plume_axis_world(x[k, 4:6], x[k, 13], x[k, 14]))
    @expression(model, plume_away_from_tower_margin[k=1:N], tower_normal' * plume_axis_world[k])
    @constraint(model, engine_plume_away_from_tower[k=1:N],
        plume_away_from_tower_margin[k] ==
        hp[k, plume_away_from_tower_index] -
        hn[k, plume_away_from_tower_index]
    )

    # Aggregate violation used in the complementarity pair. The `≥ 0` bound is
    # trivially satisfied (each hn ≥ 0); the constraint exists so total_hn is a
    # named model expression that MPCC can complement against gp.
    @constraint(model, total_hn[k=1:N],
        sum(hn[k, j] for j=1:n_state_triggered_constraints) >= 0
    )

    # Complementarity pair: gp ⟂ total_hn (below-trigger ⟂ constraint-violation).
    mpcc = MPCC(model, (model[:gp], model[:total_hn]))
    inds = var_inds(model)
    params = (; N, T, Δt, x̄, ū, m, g, l, J, catch_altitude, trigger_altitude,
                tower_normal, tower_y, arm_half_width, arm_opening_angle,
                catch_A, catch_b,
                thrust_min, thrust_max, gimbal_max, gimbal_rate_max, ϕ_max, ω_max)
    
    return (; mpcc, model, inds, params)
end

register(rocket_catch_3d, :robotics)

# problem = rocket_catch_3d(N=40, T=4.0);

# ##

# solver = Marble.Solver()
# Marble.setup!(solver, problem.mpcc; verbosity = 1)
# results = Marble.solve!(solver)

# z = Marble.z(results)