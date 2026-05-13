function to_mpcc(model::JuMP.Model)
    model_copy = copy(model)
    sos1_cons = all_constraints(model_copy, Vector{VariableRef}, MOI.SOS1{Float64})
    
    ind_vcc1 = vec(map(c -> JuMP.index(constraint_object(c).func[1]).value, sos1_cons))
    ind_vcc2 = vec(map(c -> JuMP.index(constraint_object(c).func[2]).value, sos1_cons))

    for c in sos1_cons
        delete(model_copy, c)
    end

    nlp = MathOptNLPModel(model_copy)
    @assert nlp.meta.nnln == 0 "Expected no nonlinear constraints after removing SOS1 constraints"

    nlp, ind_vcc1, ind_vcc2
end

function to_jump(ampl::AmplModel)::JuMP.Model
    function nonnegative_slack!(model, expr)
        s = @variable(model)
        set_lower_bound(s, 0.0)
        @constraint(model, s == expr)
        return s
    end

    function is_reusable_nonnegative_var(expr, lb)
        return expr isa JuMP.VariableRef &&
               JuMP.lower_bound(expr) ≈ 0
    end

    function bound_slack!(model, expr, lb, ub)
        @assert isfinite(lb) || isfinite(ub) "Need at least one finite bound"

        if isfinite(lb)
            # Complementarity side is expr - lb >= 0.
            #
            # If lb == 0 and expr is already a JuMP variable with lower bound >= 0,
            # reuse expr directly instead of creating s == expr.
            if is_reusable_nonnegative_var(expr, lb)
                # If ranged, the upper bound should already be present on the variable,
                # but this assertion helps catch inconsistencies.
                if isfinite(ub)
                    @assert JuMP.has_upper_bound(expr) "Expected upper bound on reused variable"
                    @assert JuMP.upper_bound(expr) <= ub + 1e-12
                end
                return expr
            end

            s = nonnegative_slack!(model, expr - lb)

            # If ranged, also enforce expr <= ub via s <= ub - lb.
            isfinite(ub) && @constraint(model, s <= ub - lb)

            return s
        else
            # Complementarity side is ub - expr >= 0.
            # In general this cannot reuse expr, because expr itself is not the
            # nonnegative complementarity member.
            return nonnegative_slack!(model, ub - expr)
        end
    end

    nv   = ampl.meta.nvar
    nc   = ampl.meta.ncon
    lvar = ampl.meta.lvar
    uvar = ampl.meta.uvar
    lcon = ampl.meta.lcon
    ucon = ampl.meta.ucon
    cvar = ampl.meta.cvar

    x0 = zeros(nv)

    c0 = NLPModels.obj(ampl, x0)
    g  = NLPModels.grad(ampl, x0)
    H  = NLPModels.hess(ampl, x0)
    A  = NLPModels.jac(ampl, x0)

    cc_inds = findall(!=(0), cvar)
    reg_inds = [
        i for i in 1:nc
        if cvar[i] == 0 && (isfinite(lcon[i]) || isfinite(ucon[i]))
    ]

    model = Model()

    @variable(model, x[1:nv])

    for i in eachindex(x)
        isfinite(lvar[i]) && set_lower_bound(x[i], lvar[i])
        isfinite(uvar[i]) && set_upper_bound(x[i], uvar[i])
    end

    Q = H + H' - spdiagm(0 => diag(H))

    @objective(model, Min, c0 + dot(g, x) + 0.5 * x' * Q * x)

    if !isempty(reg_inds)
        @constraint(model, lcon[reg_inds] .<= A[reg_inds, :] * x .<= ucon[reg_inds])
    end

    for i in cc_inds
        j = cvar[i]

        row_expr = A[i, :]' * x

        # Constraint-side complementarity member.
        s_c = bound_slack!(model, row_expr, lcon[i], ucon[i])

        # Variable-side complementarity member.
        # If this is 0 <= x[j], this reuses x[j] directly.
        s_v = bound_slack!(model, x[j], lvar[j], uvar[j])

        @constraint(model, [s_c, s_v] in SOS1())
    end

    return model
end