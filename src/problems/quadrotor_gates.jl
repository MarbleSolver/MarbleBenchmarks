import ForwardDiff as FD

hat(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

function L(q)
    s, v = q[1], q[2:4]
    [s -v'; v s*I(3) + hat(v)]
end

const _T = Diagonal([1.0; -ones(3)])
const _H = [zeros(1, 3); I(3)]

qtoQ(q)  = _H' * _T * L(q) * _T * L(q) * _H
Gq(q)    = L(q) * _H
Eq(q)    = BlockDiagonal([1.0*I(3), Gq(q), 1.0*I(3)])
rptoq(ϕ) = (1 / sqrt(1 + ϕ'ϕ)) * [1; ϕ]
qtorp(q) = q[2:4] / q[1]

function state_to_delta(x_full, x̄_full)
    q̄  = x̄_full[4:7]
    q   = x_full[4:7]
    δq  = L(q̄)' * q
    [x_full[1:3] - x̄_full[1:3]; qtorp(δq); x_full[8:10] - x̄_full[8:10]]
end

Rz(θ) = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1.0]

function quad_dynamics(x, u; m, g)
    q, v        = x[4:7], x[8:10]
    T_thrust, ω = u[1], u[2:4]
    [v; 0.5*L(q)*_H*ω; [0; 0; -g] + (1/m)*qtoQ(q)*[0; 0; T_thrust]]
end

function linearized_quad_dynamics(x̄, ū, Δt; m, g)
    nx, nu = 10, 4
    q0 = x̄[4:7]
    Af = FD.jacobian(x -> quad_dynamics(x, ū; m=m, g=g), x̄)
    Bf = FD.jacobian(u -> quad_dynamics(x̄, u; m=m, g=g), ū)
    cf = quad_dynamics(x̄, ū; m=m, g=g)
    sysc = ss(Af, [Bf cf], I(nx), zeros(nx, nu + 1))
    sysd = c2d(sysc, Δt, :zoh)
    Ã = Eq(q0)' * sysd.A * Eq(q0)
    B̃ = Eq(q0)' * sysd.B
    return Ã, B̃[:, 1:nu], B̃[:, end]
end

function quadrotor_gates(; N=40, T=9.0, ngates=1)
    m, g   = 1.0, 9.81
    Δt     = T / (N - 1)
    nx̃     = 9
    nu     = 4

    @assert ngates ≤ 4

    q0 = [1.0; 0; 0; 0]
    x̄  = [zeros(3); q0; zeros(3)]
    ū  = [m*g; zeros(3)]
    Ã, B̃, c̃ = linearized_quad_dynamics(x̄, ū, Δt; m=m, g=g)

    gate_positions    = [[-2.0, 2.0, 0.25], [1.0, 6.0, 1.0], [2.0, 3.0, 2.0], [-4.0, 6.0, 0.5]][1:ngates]
    gate_orientations = [Rz(deg2rad(45.0)), Rz(deg2rad(-45.0)), Rz(0.0), Rz(deg2rad(20.0))][1:ngates]
    gate_dims         = [1.0, 0.1, 1.0]

    Tmin, Tmax = 0.0, 4.0*m*g
    ωmin, ωmax = [-2.0, -2.0, -1.5], [2.0, 2.0, 1.5]
    ϕmin = deg2rad.([-20.0, -20.0, -20.0])
    ϕmax = deg2rad.([ 20.0,  20.0,  20.0])

    x0_full = [[0.0; 0.0; 0.0]; q0; zeros(3)]
    xF_full = [[0.0; 10.0; 0.0]; [1.0; 0.0; 0.0; 0.0]; zeros(3)]
    xstart  = state_to_delta(x0_full, x̄)
    xfinal  = state_to_delta(xF_full, x̄)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx̃])
    @variable(model, u[1:N-1, 1:nu])
    @variable(model, e[1:N, 1:ngates, 1:3])
    @variable(model, λ[1:N, 1:ngates] >= 0)
    @variable(model, μ[1:N, 1:ngates] >= 0)
    @variable(model, s_pos[1:N, 1:ngates, 1:3] >= 0)
    @variable(model, s_neg[1:N, 1:ngates, 1:3] >= 0)

    slack_penalty = 1.0

    @objective(model, Min,
        sum(Δt / 2 * (ū + u[k, :])' * (ū + u[k, :]) for k=1:N-1)
        + slack_penalty * Δt * sum(s_pos[k, j, d] + s_neg[k, j, d] for k=1:N, j=1:ngates, d=1:3)
    )

    @constraint(model, ic, x[1, :] .== xstart)
    @constraint(model, tc, x[N, :] .== xfinal)
    @constraint(model, λ_terminal, λ[N, :] .>= 1.0)
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ã * x[k, :] + B̃ * u[k, :] + c̃)
    @constraint(model, rp_lb[k=1:N, d=1:3], x[k, 3+d] >= ϕmin[d])
    @constraint(model, rp_ub[k=1:N, d=1:3], x[k, 3+d] <= ϕmax[d])
    @constraint(model, T_lb[k=1:N-1], ū[1] + u[k, 1] >= Tmin)
    @constraint(model, T_ub[k=1:N-1], ū[1] + u[k, 1] <= Tmax)
    @constraint(model, ω_lb[k=1:N-1], ū[2:4] + u[k, 2:4] .>= ωmin)
    @constraint(model, ω_ub[k=1:N-1], ū[2:4] + u[k, 2:4] .<= ωmax)
    @constraint(model, waypoint_dyn_ic, λ[1, :] .== μ[1, :])
    @constraint(model, waypoint_dyn[k=2:N], λ[k, :] .== λ[k-1, :] + μ[k, :])
    @constraint(model, waypoint_order[k=1:N, j=1:ngates-1], λ[k, j] >= λ[k, j+1])

    h = gate_dims ./ 2
    @expression(model, gate_error[k=1:N, j=1:ngates], gate_orientations[j]' * (x̄[1:3] + x[k, 1:3] - gate_positions[j]))
    @constraint(model, gate_upper[k=1:N, j=1:ngates, d=1:3], gate_error[k,j][d] <= h[d] + s_pos[k,j,d])
    @constraint(model, gate_lower[k=1:N, j=1:ngates, d=1:3], gate_error[k,j][d] >= -h[d] - s_neg[k,j,d])
    @constraint(model, gate_violation[k=1:N, j=1:ngates], sum(s_pos[k,j,:] + s_neg[k,j,:]) >= 0)

    complementarities = [
        (var_con_complementarities(model, model[:μ], model[:gate_violation]), (:var, :con)),
    ]

    ind_cc1  = reduce(vcat, first.(p) for (p, _) in complementarities)
    ind_cc2  = reduce(vcat, last.(p)  for (p, _) in complementarities)
    cc_types = reduce(vcat, fill(t, length(p)) for (p, t) in complementarities)

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, Δt, x̄, ū, ngates, gate_positions, gate_orientations, gate_dims))
end

register(quadrotor_gates, :robotics)
