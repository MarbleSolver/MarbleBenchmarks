# ===========================================================================
# Minimal 1-D gate problem — EXACT complementarity formulation (no slacks).
#
# A robot moves in 1-D and must visit each gate j (the interval
# [xgate[j]-h, xgate[j]+h]) at some timestep, minimizing control effort.
#
# Unlike the slack version, the per-gate "trigger" here is pinned EXACTLY by
# complementarity rather than being a free lower-bounded slack.  Using the
# identity
#
#       p = max(0, f)   ⟺   0 ≤ p ⊥ (p − f) ≥ 0,
#
# the two one-sided violations are charted exactly:
#
#       over[k,j]  = max(0, x[k] − (c+h))     (how far ABOVE gate j)
#       under[k,j] = max(0, (c−h) − x[k])     (how far BELOW gate j)
#
# so the gate trigger  g[k,j] = over[k,j] + under[k,j]  is EXACTLY 0 when the
# robot is inside gate j and > 0 otherwise (the two sides are mutually
# exclusive, so g is exactly the distance outside the gate interval).
#
# Gate credit μ[k,j] ≥ 0 may then be earned only inside the gate:
#
#       0 ≤ μ[k,j] ⊥ g[k,j] ≥ 0      ⇒   μ[k,j] > 0 ⟹ trigger = 0 ⟹ inside.
#
# Cumulative credit λ[k,j] = Σ_{i≤k} μ[i,j] with λ[N,j] ≥ 1 forces every gate to
# be visited; the optional `in_order` chain forces visiting them in index order.
# ===========================================================================

function minimal_gate_mpcc_cc(; N=4, xgate=[10.0, -10.0], gate_half_width=0.5, x0=-2.0, in_order=false)
    h = gate_half_width  # gate half-width; gate j is the interval [xgate[j]-h, xgate[j]+h]

    ngates = length(xgate)

    model = JuMP.Model()

    @variable(model, x[1:N])              # 1-D position
    @variable(model, u[1:N-1])            # control input
    @variable(model, μ[1:N, 1:ngates] >= 0)      # gate-credit increment
    @variable(model, over[1:N, 1:ngates]  >= 0)  # overshoot above gate  (= max(0, x-(c+h)))
    @variable(model, under[1:N, 1:ngates] >= 0)  # undershoot below gate (= max(0, (c-h)-x))

    @objective(model, Min, sum(u[k]^2 for k in 1:N-1))

    @constraint(model, x[1] == x0)                          # start position
    @constraint(model, dyn[k=1:N-1], x[k+1] == x[k] + u[k]) # single-integrator dynamics

    # cumulative credit; every gate must be credited (visited) at least once
    @expression(model, λ[k=1:N, j=1:ngates], sum(μ[i, j] for i in 1:k))
    @constraint(model, [j=1:ngates], λ[N, j] == 1.0)
    if in_order
        @constraint(model, inorder[k=1:N, j=1:ngates-1], λ[k, j] >= λ[k, j+1])
    end

    # Exact trigger pins:  0 ≤ over  ⊥ (over  − (x-(c+h))) ≥ 0  ⇒ over  = max(0, x-(c+h))
    #                      0 ≤ under ⊥ (under − ((c-h)-x)) ≥ 0  ⇒ under = max(0, (c-h)-x)
    @constraint(model, relu_hi[k=1:N, j=1:ngates], over[k, j]  - (x[k] - (xgate[j] + h)) >= 0)
    @constraint(model, relu_lo[k=1:N, j=1:ngates], under[k, j] - ((xgate[j] - h) - x[k]) >= 0)

    # Credit only inside:  0 ≤ μ ⊥ (over + under) ≥ 0    (trigger g = over + under)
    @constraint(model, gate_out[k=1:N, j=1:ngates], over[k, j] + under[k, j] >= 0)

    return MPCCJuMP(model,
        (over,  relu_hi),     # over  ⊥ relu residual  ⇒ over  = max(0, x-(c+h))
        (under, relu_lo),     # under ⊥ relu residual  ⇒ under = max(0, (c-h)-x)
        (μ,     gate_out),    # credit ⊥ trigger       ⇒ credit only inside the gate
    )
end

register(minimal_gate_mpcc_cc, :debug)
