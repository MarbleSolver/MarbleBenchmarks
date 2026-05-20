using JuMP
using LinearAlgebra
using MarbleBenchmarks
using NLPModelsJuMP

function quadrotor_gates_double_integrator(; 
    N=40, 
    T=9.0, 
    ngates=1,
    accel_bound=Inf,
)
    @assert N >= 2
    @assert ngates ≤ 4

    Rz(θ) = [
        cos(θ) -sin(θ) 0.0
        sin(θ)  cos(θ) 0.0
        0.0     0.0    1.0
    ]


    Δt = T / (N - 1)
    nx = 6
    nu = 3

    gate_positions = [
        [-2.0, 2.0, 0.25],
        [ 1.0, 6.0, 1.0],
        [ 2.0, 3.0, 2.0],
        [-4.0, 6.0, 0.5],
    ][1:ngates]

    gate_orientations = [
        Rz(deg2rad(45.0)),
        Rz(deg2rad(-45.0)),
        Rz(0.0),
        Rz(deg2rad(20.0)),
    ][1:ngates]

    gate_dims = [1.0, 0.1, 1.0]
    hgate = gate_dims ./ 2

    p0 = [0.0, 0.0, 0.0]
    v0 = [0.0, 0.0, 0.0]

    pF = [0.0, 10.0, 0.0]
    vF = [0.0, 0.0, 0.0]

    xstart = [p0; v0]
    xfinal = [pF; vF]

    model = JuMP.Model()

    # State: x[k, 1:3] = position, x[k, 4:6] = velocity
    @variable(model, x[1:N, 1:nx])

    # Control: acceleration
    @variable(model, u[1:N-1, 1:nu])

    # Gate credit variables
    @variable(model, λ[1:N, 1:ngates] >= 0)
    @variable(model, μ[1:N, 1:ngates] >= 0)

    # Positive and negative gate-violation slacks
    #
    # s_pos[k,j,d] = max(0, e_d - h_d)
    # s_neg[k,j,d] = max(0, -h_d - e_d)
    #
    # enforced below through KKT conditions.
    @variable(model, s_pos[1:N, 1:ngates, 1:3] >= 0)
    @variable(model, s_neg[1:N, 1:ngates, 1:3] >= 0)

    # Epigraph residuals:
    #
    # eta_pos = s_pos - (e - h) >= 0
    # eta_neg = s_neg - (-h - e) >= 0
    @variable(model, eta_pos[1:N, 1:ngates, 1:3] >= 0)
    @variable(model, eta_neg[1:N, 1:ngates, 1:3] >= 0)

    # Total minimal violation for each gate at each timestep.
    #
    # μ[k,j] ⟂ gate_viol[k,j]
    @variable(model, gate_viol[1:N, 1:ngates] >= 0)

    # KKT multipliers for hidden minimization of s_pos and s_neg.
    #
    # For s_pos:
    #   min s_pos
    #   s.t. s_pos >= 0
    #        s_pos >= e - h
    #
    # For s_neg:
    #   min s_neg
    #   s.t. s_neg >= 0
    #        s_neg >= -h - e
    @variable(model, α_pos0[1:N, 1:ngates, 1:3] >= 0)
    @variable(model, α_pos1[1:N, 1:ngates, 1:3] >= 0)

    @variable(model, α_neg0[1:N, 1:ngates, 1:3] >= 0)
    @variable(model, α_neg1[1:N, 1:ngates, 1:3] >= 0)

    @objective(model, Min,
        sum(Δt / 2 * sum(u[k, d]^2 for d in 1:3) for k in 1:N-1)
    )

    # Initial / terminal constraints
    @constraint(model, ic[d=1:nx], x[1, d] == xstart[d])
    @constraint(model, tc[d=1:nx], x[N, d] == xfinal[d])

    # Double-integrator dynamics
    @constraint(model, dynamics_pos[k=1:N-1, d=1:3],
        x[k+1, d] == x[k, d] + Δt * x[k, 3+d] + 0.5 * Δt^2 * u[k, d]
    )

    @constraint(model, dynamics_vel[k=1:N-1, d=1:3],
        x[k+1, 3+d] == x[k, 3+d] + Δt * u[k, d]
    )

    if isfinite(accel_bound)
        @constraint(model, accel_lb[k=1:N-1, d=1:3], u[k, d] >= -accel_bound)
        @constraint(model, accel_ub[k=1:N-1, d=1:3], u[k, d] <=  accel_bound)
    end

    # Gate-credit dynamics
    @constraint(model, waypoint_dyn_ic[j=1:ngates],
        λ[1, j] == μ[1, j]
    )

    @constraint(model, waypoint_dyn[k=2:N, j=1:ngates],
        λ[k, j] == λ[k-1, j] + μ[k, j]
    )

    # I strongly recommend equality here, not >=.
    # With >=, λ and μ are unnecessarily unbounded / degenerate.
    @constraint(model, λ_terminal[j=1:ngates],
        λ[N, j] == 1.0
    )

    if ngates > 1
        @constraint(model, waypoint_order[k=1:N, j=1:ngates-1],
            λ[k, j] >= λ[k, j+1]
        )
    end

    # Gate-frame position error:
    #
    # e[k,j] = R_j' * (p_k - p_gate_j)
    #
    # Component d is:
    # e_d = sum_r R_j[r,d] * (p_r - p_gate_r)
    @expression(model, gate_error[k=1:N, j=1:ngates, d=1:3],
        sum(gate_orientations[j][r, d] * (x[k, r] - gate_positions[j][r]) for r in 1:3)
    )

    # Minimal-slack epigraph residual definitions
    @constraint(model, eta_pos_def[k=1:N, j=1:ngates, d=1:3],
        eta_pos[k, j, d] == s_pos[k, j, d] - (gate_error[k, j, d] - hgate[d])
    )

    @constraint(model, eta_neg_def[k=1:N, j=1:ngates, d=1:3],
        eta_neg[k, j, d] == s_neg[k, j, d] - (-hgate[d] - gate_error[k, j, d])
    )

    # Hidden KKT stationarity for:
    #
    # s_pos = argmin_s s
    # subject to s >= 0, s >= e - h
    #
    # and similarly for s_neg.
    @constraint(model, s_pos_stationarity[k=1:N, j=1:ngates, d=1:3],
        α_pos0[k, j, d] + α_pos1[k, j, d] == 1.0
    )

    @constraint(model, s_neg_stationarity[k=1:N, j=1:ngates, d=1:3],
        α_neg0[k, j, d] + α_neg1[k, j, d] == 1.0
    )

    # Total gate violation is the sum of minimal per-face violations.
    @constraint(model, gate_viol_def[k=1:N, j=1:ngates],
        gate_viol[k, j] == sum(s_pos[k, j, d] + s_neg[k, j, d] for d in 1:3)
    )

    # Complementarity blocks
    #
    # 1. Gate credit only allowed when total minimal violation is zero:
    #       μ[k,j] ⟂ gate_viol[k,j]
    #
    # 2. KKT complementarity for s_pos minimality:
    #       α_pos0 ⟂ s_pos
    #       α_pos1 ⟂ eta_pos
    #
    # 3. KKT complementarity for s_neg minimality:
    #       α_neg0 ⟂ s_neg
    #       α_neg1 ⟂ eta_neg
    comp_blocks = [
        (var_var_complementarities(model, model[:μ],      model[:gate_viol]), (:var, :var)),

        (var_var_complementarities(model, model[:α_pos0], model[:s_pos]),     (:var, :var)),
        (var_var_complementarities(model, model[:α_pos1], model[:eta_pos]),   (:var, :var)),

        (var_var_complementarities(model, model[:α_neg0], model[:s_neg]),     (:var, :var)),
        (var_var_complementarities(model, model[:α_neg1], model[:eta_neg]),   (:var, :var)),
    ]

    ind_cc1 = reduce(vcat, first.(p) for (p, _) in comp_blocks)
    ind_cc2 = reduce(vcat, last.(p)  for (p, _) in comp_blocks)
    cc_types = reduce(vcat, fill(t, length(p)) for (p, t) in comp_blocks)

    return (; 
        model, 
        ind_cc1, 
        ind_cc2, 
        cc_types,
        inds = var_inds(model),
        params = (; 
            N, 
            T, 
            Δt, 
            ngates, 
            gate_positions, 
            gate_orientations, 
            gate_dims,
            hgate,
        ),
    )
end

register(quadrotor_gates_double_integrator, :robotics)