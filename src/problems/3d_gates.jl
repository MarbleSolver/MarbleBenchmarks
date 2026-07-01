function three_d_gates_fewer_comps(;
        N=50,
        T=9.0,
        ngates=2,
        max_gate_violation=20.0,
        gate_violation_weight=1e-4,
    )
    Δt = T / (N - 1)
    nx, nu = 6, 3

    @assert ngates ≤ 4

    gate_positions    = [[-2.0, 2.0, 0.25], [1.0, 6.0, 1.0], [2.0, 3.0, 2.0], [-4.0, 6.0, 0.5]][1:ngates]
    gate_orientations = [Rz(deg2rad(45.0)), Rz(deg2rad(-45.0)), Rz(0.0), Rz(deg2rad(20.0))][1:ngates]
    gate_dims         = [1.0, 0.1, 1.0]
    h = gate_dims ./ 2

    x0 = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    xF = [0.0; 10.0; 0.0; 0.0; 0.0; 0.0]

    model = JuMP.Model()

    # x[k, :] = [px, py, pz, vx, vy, vz].
    @variable(model, x[1:N, 1:nx])
    # u[k, :] is acceleration input for the double integrator.
    @variable(model, u[1:N-1, 1:nu])
    # μ[k, j] is gate progress. It may be positive only when gate_violation is 0.
    @variable(model, μ[1:N, 1:ngates] >= 0)
    @variable(model, 0 <= gate_violation[1:N, 1:ngates])

    @objective(model, Min,
        sum((Δt / 2 * u[k, :]' * u[k, :]) for k=1:N-1)
        # A small regularizer removes the unbounded "make violation huge" direction
        # that is invisible to the control objective but poisonous for Marble.
        + sum((Δt / 2) * gate_violation_weight * gate_violation[k, :]'*gate_violation[k, :] for k=1:N)
    )

    @constraint(model, ic, x[1, :] .== x0)
    @constraint(model, tc, x[N, :] .== xF)
    @constraint(model, dynamics[k=1:N-1],
        x[k+1, :] .== [
            x[k, 1] + Δt * x[k+1, 4];
            x[k, 2] + Δt * x[k+1, 5];
            x[k, 3] + Δt * x[k+1, 6];
            x[k, 4] + Δt * u[k, 1];
            x[k, 5] + Δt * u[k, 2];
            x[k, 6] + Δt * u[k, 3];
        ]
    )

    @expression(model, progress[k=1:N, j=1:ngates], sum(μ[t, j] for t=1:k))
    @constraint(model, progress_terminal[j=1:ngates], progress[N, j] >= 1.0)
    @constraint(model, progress_upper[k=1:N, j=1:ngates], progress[k, j] <= 1.0)
    @constraint(model, progress_order[k=1:N, j=1:ngates-1], progress[k, j] >= progress[k, j+1])

    @expression(model, gate_error[k=1:N, j=1:ngates],
        gate_orientations[j]' * (x[k, 1:3] - gate_positions[j])
    )

    # This is a one-sided certificate: gate_violation == 0 implies the robot is
    # inside every gate boundary. We do not need gate_violation to equal the true
    # max violation away from progress nodes.
    @constraint(model, gate_violation_upper_excess[k=1:N, j=1:ngates, d=1:3],
        gate_violation[k, j] - (gate_error[k, j][d] - h[d]) >= 0
    )
    @constraint(model, gate_violation_lower_excess[k=1:N, j=1:ngates, d=1:3],
        gate_violation[k, j] - (-gate_error[k, j][d] - h[d]) >= 0
    )

    return MPCCJuMP(model,
        (model[:μ], model[:gate_violation]);
        params=(; N, T, Δt, ngates, gate_positions, gate_orientations,
                 gate_dims, max_gate_violation, gate_violation_weight))
end

register(three_d_gates_fewer_comps, Symbol("3d_gates_fewer_comps"), :robotics)
