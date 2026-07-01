function minimal_gate_mpcc(; N=4, xgate=[10.0, -10.0], gate_half_width=0.5, x0=-2.0, in_order=false)
    ngates = length(xgate)

    model = JuMP.Model()

    @variable(model, x[1:N])          # 1D position
    @variable(model, u[1:N-1])        # control input
    @variable(model, μ[1:N, 1:ngates] >= 0)     # gate-credit increment
    @variable(model, s[1:N, 1:ngates] >= 0)     # gate violation slack

    @objective(model, Min,
        sum(u[k]^2 for k in 1:N-1)
    )

    @constraint(model, x[1] == x0)                                  # start outside gate
    @constraint(model, dyn[k=1:N-1], x[k+1] == x[k] + u[k])

    @expression(model, λ[k=1:N, j=1:ngates], sum(μ[i, j] for i in 1:k))  # λ[k] = sum of μ[1:k]
    @constraint(model, [j=1:ngates], λ[N, j] >= 1.0)

    if in_order
        @constraint(model, inorder[k=1:N, j=1:ngates-1], λ[k, j] >= λ[k, j+1])
    end
    
    # If x ∈ [x_gate - h, x_gate + h] then we have:
    # x[k] - (xgate + h) is negative, (xgate - h) - x[k] is negative, so s can be 0
    # and μ can be nonzero. If x > xgate + h, then x[k] - (xgate + h) is positive, so s > 0
    # and if x < xgate - h, then (xgate - h) - x[k] is positive, so s > 0
    # so μ must be 0 in both cases.

    @constraint(model, gate_ub[k=1:N, j=1:ngates], x[k] - (xgate[j] + gate_half_width) <= s[k, j])
    @constraint(model, gate_lb[k=1:N, j=1:ngates], (xgate[j] - gate_half_width) - x[k] <= s[k, j])

    return MPCCJuMP(model, (model[:μ], model[:s]))
end

register(minimal_gate_mpcc, :debug)
