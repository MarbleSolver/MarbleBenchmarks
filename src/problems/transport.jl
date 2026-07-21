# Cart transport (CRISP `Transp` example, arXiv:2502.01055).
function transport(; N=200, T=3.98)
    m1, m2 = 1.0, 2.0      # cargo mass, cart mass
    μ      = 0.2           # friction coefficient
    g      = 9.81
    l      = 1.0           # max cargo/cart separation
    dt     = T / (N - 1)
    nx     = 4             # state x = [x1, x2, ẋ1, ẋ2]

    friction_bound = μ * m1 * g

    # x = [cargo pos, cart pos, cargo vel, cart vel]
    x0 = [ 3.5,  3.0, -4.0, -4.0]
    xF = [-0.5,  0.0, -2.0, -2.0]
    Q_terminal = [1e5, 1e5, 10.0, 10.0]     # terminal tracking weights
    R_control  = 1e-4                     # control effort weight on [f, u]

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx])
    # f is the friction force on the cargo; u is the force applied to the cart
    @variable(model, f[1:N-1])
    @variable(model, u[1:N-1])
    # v, w split the cargo/cart relative velocity into its +x and -x parts
    @variable(model, v[1:N-1] >= 0)
    @variable(model, w[1:N-1] >= 0)

    # Terminal tracking cost plus control effort (no time scaling, per CRISP).
    @objective(model, Min,
        sum(Q_terminal[j] * (x[N, j] - xF[j])^2 for j in 1:nx)
        + sum(R_control * (f[k]^2 + u[k]^2) for k in 1:N-1)
    )

    @constraint(model, initial_state, x[1, :] .== x0)

    cargo_pos(k) = x[k, 1]
    cart_pos(k)  = x[k, 2]
    cargo_vel(k) = x[k, 3]
    cart_vel(k)  = x[k, 4]

    # Semi-implicit Euler dynamics: ẍ1 = f/m1, ẍ2 = (u - f)/m2.
    @expression(model, continuous_dynamics[k=1:N-1],
        [
            cargo_vel(k+1);
            cart_vel(k+1);
            f[k] / m1;
            (u[k] - f[k]) / m2;
        ]
    )
    @constraint(model, dynamics[k=1:N-1],
        x[k+1, :] .== x[k, :] + dt .* continuous_dynamics[k])

    # Sliding velocity split: ẋ1 - ẋ2 = v - w.
    @constraint(model, sliding_velocity[k=1:N-1],
        cargo_vel(k) - cart_vel(k) - v[k] + w[k] == 0)

    # Friction cone -μ m1 g <= f <= μ m1 g, as two nonnegative slacks.
    @constraint(model, friction_upper[k=1:N-1], friction_bound - f[k] >= 0)
    @constraint(model, friction_lower[k=1:N-1], f[k] + friction_bound >= 0)

    # Combined nonnegative slack v is complementary to: the -x slide part w plus
    # the lower friction slack f + μ m1 g.
    @constraint(model, v_complement[k=1:N-1], w[k] + f[k] + friction_bound >= 0)

    # Length constraint on the cargo/cart separation: |x1 - x2| <= l.
    @constraint(model, separation_upper[k=1:N], l - (cargo_pos(k) - cart_pos(k)) >= 0)
    @constraint(model, separation_lower[k=1:N], (cargo_pos(k) - cart_pos(k)) + l >= 0)

    # Maximum-dissipation complementarity. Because v, w, and (f + μ m1 g) are each
    # nonnegative, the two conditions v ⊥ w and v ⊥ (f + μ m1 g) are equivalent to
    # a single pairing of v against their sum:
    #   v ⊥ (w + f + μ m1 g) : no simultaneous +/- slide, +x slide pins friction low
    #   w ⊥ (μ m1 g - f)     : sliding in -x pins friction to its upper bound
    mpcc = MPCC(model,
        (model[:v], model[:v_complement]),
        (model[:w], model[:friction_upper]),
    )
    inds = var_inds(model)
    params = (; N, T, dt, m1, m2, μ, g, l, x0, xF)

    return (; mpcc, model, inds, params)
end

register(transport)
