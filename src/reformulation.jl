function mpcc_to_sos1(mpcc::Marble.MPCC)::JuMP.Model
    (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = mpcc_to_marble(mpcc)

    # Create primal vars and Model
    model = JuMP.Model()
    n_var = length(q)
    n_cc = size(L, 1)

    @variable(model, x[1:n_var])
    @objective(model, Min, 0.5 * x' * Q * x + q' * x + c0)
    @constraint(model, J_eq * x + b_eq .== 0)
    @constraint(model, J_ineq * x + b_ineq .>= 0)

    # Each complementarity pair j is `0 <= (L x + l)_j ⊥ (R x + r)_j >= 0`. We
    # rewrite it as an SOS1 over the two endpoints. SOS1 only accepts variable endpoints,
    # so we introduce a slack variable for the endpoint that is not a variable.
    cc1_kind = first.(mpcc.cc_types)
    cc2_kind = last.(mpcc.cc_types)
    cc1_res = L * x .+ l
    cc2_res = R * x .+ r

    # SOS1 member variable for one endpoint, restoring its dropped `>= 0` bound
    function sos1_member(kind, var_ind, residual)
        if kind === :var
            @constraint(model, residual >= 0)               # restore 0 <= endpoint
            return x[var_ind]
        else
            slack = @variable(model, lower_bound = 0.0)      # 0 <= slack == endpoint
            @constraint(model, slack == residual)
            return slack
        end
    end

    for j in 1:n_cc
        a = sos1_member(cc1_kind[j], mpcc.ind_cc1[j], cc1_res[j])
        b = sos1_member(cc2_kind[j], mpcc.ind_cc2[j], cc2_res[j])
        @constraint(model, [a, b] in JuMP.SOS1())
    end

    return model
end