import ForwardDiff as FD

# Kinematic (rear-axle) bicycle model.
#   State  x = [px, py, θ, v]   position, heading, forward speed
#   Control u = [a, δ]          longitudinal acceleration, steering angle
#   L is the wheelbase.
function bicycle_kinematics_2d(x, u; L)
    θ, v = x[3], x[4]
    a, δ = u
    [v * cos(θ); v * sin(θ); v * tan(δ) / L; a]
end

# Discrete-time dynamics via ZOH about (x̄, ū): x_{k+1} = Ad·x_k + Bd·u_k + cd.
function linearized_bicycle_kinematics_2d(x̄, ū, Δt; L)
    nx, nu = 4, 2
    A = FD.jacobian(x -> bicycle_kinematics_2d(x, ū; L), x̄)
    B = FD.jacobian(u -> bicycle_kinematics_2d(x̄, u; L), ū)
    c = bicycle_kinematics_2d(x̄, ū; L)
    sysd = c2d(ss(A, [B c], I(nx), zeros(nx, nu + 1)), Δt, :zoh)
    return sysd.A, sysd.B[:, 1:nu], sysd.B[:, end]
end

# kph → m/s
kph2mps(v) = v / 3.6

function lane_passing_2d(; N=40, T=5.0, L=2.5,     # L: wheelbase [m]
    lane_width_m=3.7,
    cruiser_kph=45,
    x0 = [-15.0, -1e-3, 0.0, 40 / 3.6],
    θ_max_deg = 10.0, δ_max_deg = 12.0,   # linearization bounds
    v_min_kph = 30.0, v_max_kph = 80.0,  # passer speed envelope
)
    Δt = T / (N - 1)
    nx, nu = 4, 2

    # Cruiser constant speed [m/s]
    cruiser_mps = kph2mps(cruiser_kph)
    lane_width = lane_width_m           # [m]

    x̄ = [0.0, 0.0, 0.0, cruiser_mps]
    ū = [0.0, 0.0]
    Ad, Bd, cd = linearized_bicycle_kinematics_2d(x̄, ū, Δt; L=L)

    model = JuMP.Model()

    # Set up the polytopes.  At θ = 0 the bicycle travels along coordinate 1, so
    # coordinate 1 is longitudinal (car length) and coordinate 2 is lateral (width).
    w_cruiser = lane_width_m # m, width
    h_cruiser = 9.0     # m, length
    A_cruiser = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
    b_cruiser = [h_cruiser; h_cruiser; w_cruiser; w_cruiser] ./ 2

    w_passer = 2.1      # m, width
    h_passer = 5.0      # m, length
    cθ, sθ = cos(deg2rad(θ_max_deg)), sin(deg2rad(θ_max_deg))

    h_passer_eff = h_passer * cθ + w_passer * sθ
    w_passer_eff = h_passer * sθ + w_passer * cθ

    A_passer = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
    b_passer = [h_passer_eff; h_passer_eff; w_passer_eff; w_passer_eff] ./ 2

    # Normalize the polytopes to better scale the LP multipliers λ
    Ã_cruiser = A_cruiser ./ b_cruiser        # rows: ±2/h_c, ±2/w_c
    Ã_passer  = A_passer  ./ b_passer         # rows: ±2/h_p_eff, ±2/w_p_eff
    b̃ = ones(4)

    G = [-Ã_cruiser b̃
         -Ã_passer  b̃]
    h(c1, c2) = [Ã_cruiser * c1; Ã_passer * c2]

    e3 = [0.0, 0.0, 1.0]   # = e_{n+1}, picks out α in z = [x; α]

    @variable(model, x[1:N, 1:nx])
    @variable(model, u[1:N-1, 1:nu])

    @variable(model, α[1:N] >= 1.0) # growth distance between the two polytopes
    @variable(model, lp_λ[1:N, 1:size(G, 1)] >= 0)
    @variable(model, lp_x[1:N, 1:2])

    # Objective fn: be in the center lane at the end, min effort.  In SI units the
    # 1/max² weights make each control's contribution O(1) at its bound: a_max is
    # O(1) m/s² so R[1,1] = 1/a_max² is O(0.1)
    a_max = 3.0         # m/s²
    R = Diagonal([1.0 / a_max^2, 1.0 / deg2rad(δ_max_deg)^2])

    # Running state cost.  Each state is normalized by its characteristic scale
    # (1/scale²), so a *priority* of 1.0 means "sitting at that scale costs the
    # same as saturating one control".  This keeps the weights comparable even
    # though the states live on very different numeric scales (py ~ metres,
    # θ ~ 0.1 rad, v ~ m/s) — with raw equal weights the heading term (θ² ~ 0.03)
    # is negligible while the speed term (v² ~ 150) dominates.
    py_scale = lane_width                            # one lane-width of drift [m]
    θ_scale  = deg2rad(θ_max_deg)                    # heading at the lin. bound [rad]
    v_scale  = kph2mps(v_max_kph) - cruiser_mps      # top of the speed envelope [m/s]

    # O(1) priority knobs — tune these, not the raw magnitudes.
    w_lat   = 0.01    # ↑ pulls the passer back toward the centre lane sooner
    w_head  = 2.0    # ↑ keeps the heading small → gentler, wider lane changes
    w_speed = 0.1    # ↑ keeps speed near the cruiser (slower, tighter pass)

    Q = Diagonal([
        0.0,                     # px: longitudinal position is free
        w_lat   / py_scale^2,    # py: lateral deviation from the centre lane
        w_head  / θ_scale^2,     # θ:  heading (gentle turn-out / turn-in)
        w_speed / v_scale^2,     # v:  speed deviation from the cruiser
    ])

    @objective(model, Min,
        sum(1/2 * Δt * (x[i, :]' * Q * x[i, :]) for i in 1:N) +
        sum(1/2 * Δt * (u[i, :]' * R * u[i, :]) for i in 1:N-1)
    )

    # Linearization bounds |θ| ≤ 15° and |δ| ≤ 15° (≈ 0.2618 rad)
    @constraint(model, θ_bounds, -deg2rad(θ_max_deg) .<= x[:, 3] .<= deg2rad(θ_max_deg))
    @constraint(model, δ_bounds, -deg2rad(δ_max_deg) .<= u[:, 2] .<= deg2rad(δ_max_deg))

    # Speed envelope. x[:, 4] is speed deviation from the cruiser (error coords),
    # so the true speed is cruiser_mps + x[k, 4].
    v_min, v_max = kph2mps(v_min_kph), kph2mps(v_max_kph)
    @constraint(model, v_bounds, v_min .<= cruiser_mps .+ x[:, 4] .<= v_max)

    # Dynamics
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ad * x[k, :] + Bd * u[k, :] + cd)

    # Initial condition
    @constraint(model, ic, x[1, :] .== (x0 - x̄))   # 30 m behind, at cruiser speed

    # Be in front of the cruiser at the end, center lane, facing forward
    @constraint(model, terminal_ahead,
        x[N, 1] >= cruiser_mps * Δt * (N - 1) + (h_cruiser + h_passer) / 2 + 10.0)
    @constraint(model, terminal_lane, x[N, 2] == 0.0)
    @constraint(model, terminal_heading, x[N, 3] == 0.0)

    # Stay in the lane boundary for all timesteps
    @constraint(model, road_lb, x[:, 2] .>= -3/2 * lane_width + w_passer_eff/2)
    @constraint(model, road_ub, x[:, 2] .<=  3/2 * lane_width - w_passer_eff/2)

    # Growth distance between two polytopes.
    #
    #   α* = min_{x, α} α
    #        s.t. A1 x ≤ α b1 + A1 c1
    #             A2 x ≤ α b2 + A2 c2
    #
    # A1 x ≤ b1 and A2 x ≤ b2 describe the polytopes in their own body frames,
    # centered at the origin (so b1, b2 ≥ 0, making the α-scaled family nested).
    # c1, c2 translate them into the world. α scales each polytope about its own
    # center; the polytopes intersect iff α* ≤ 1, so non-intersection is α* ≥ 1.
    #
    # We can alternatively write the problem as:
    #
    #   α* = min_{x, α} α
    #        s.t. G [x; α] + h ≥ 0
    # with G = [-A1 b1; -A2 b2] and h = [A1 c1; A2 c2]
    # The KKT conditions are:
    # e_{n+1} - G' * λ = 0
    # 0 ≤ G [x; α] + h ⟂ λ ≥ 0
    r_cruiser(i) = [cruiser_mps * Δt * (i - 1), 0.0]
    m = size(G, 1)
    @expression(model, lp_kkt_expr[i=1:N],
        G * [lp_x[i, :]; α[i]] + h(r_cruiser(i), x[i, 1:2]))

    # Regularization of the lower-level witness point.  For two boxes the
    # growth-distance LP has a NON-UNIQUE witness whenever contact is edge–edge
    # (the parallel long sides during a pass): any point along the contact segment
    # is optimal, so the lower-level KKT Jacobian is singular and the inner AL
    # subproblem fails to converge.  Adding (ε/2)‖lp_x − x_mid‖² makes the lower
    # level strictly convex in lp_x, so the witness (and hence λ) is unique and
    # the KKT system is well posed.  x_mid is the midpoint of the two body centers,
    # so the bias is symmetric and vanishes as ε → 0.  The stationarity gradient
    # picks up the ε term in its lp_x components: e3 + ε[lp_x − x_mid; 0] − Gᵀλ = 0.
    ε_witness = 1e-2
    x_mid(i) = (r_cruiser(i) + x[i, 1:2]) / 2
    @constraint(model, lp_stationarity[i=1:N],
        e3 + [ε_witness * (lp_x[i, :] - x_mid(i)); 0.0] - G' * lp_λ[i, :] .== 0)

    # Scalar-indexed so each entry is a single ConstraintRef, matching lp_λ[i, j]
    # elementwise for the complementarity block (both vec column-major over (i, j)).
    @constraint(model, lp_kkt_lhs[i=1:N, j=1:m], lp_kkt_expr[i][j] >= 0)

    # Build problem
    mpcc = MPCC(model, (model[:lp_kkt_lhs], model[:lp_λ]))
    inds = var_inds(model)
    params = (;
        N, T, Δt, nx, nu, x̄, L, lane_width, cruiser_mps, w_cruiser, h_cruiser, w_passer, h_passer,
        A_cruiser, b_cruiser, A_passer, b_passer,
    )

    return (; mpcc, model, inds, params)
end

register(lane_passing_2d, :robotics)
