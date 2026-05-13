using Marble
using JSOSolvers
using NLPModels, NLPModelsJuMP, JuMP, AmplNLReader
using LinearAlgebra
using MPCCModels
using Printf
using Clarabel

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MPCCModels.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

function try_project_solution(data::MarbleData; x::AbstractVector, comp_tol::Real = 1e-2)
    n_var = length(data.q)
    n_cc  = length(data.l)

    sL = data.L * x .+ data.l
    sR = data.R * x .+ data.r

    # Only attempt rounding when all CC pair products are near zero
    n_cc > 0 && norm(sL .* sR, Inf) > comp_tol && return nothing

    # For each CC pair, the smaller-magnitude side is rounded to 0 (active),
    # and the larger side is kept free with a >= 0 constraint.
    active_L = abs.(sL) .<= abs.(sR)    # true  -> L*x+l = 0,  R*x+r >= 0
                                        # false -> R*x+r = 0,  L*x+l >= 0

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, xv[1:n_var])

    @objective(model, Min, 0.5 * dot(xv, data.Q, xv) + dot(data.q, xv))

    # Original equality constraints: J_eq * x + b_eq = 0
    !isempty(data.b_eq) &&
        @constraint(model, data.J_eq * xv .+ data.b_eq .== 0)

    # Original inequality constraints: J_ineq * x + b_ineq >= 0
    !isempty(data.b_ineq) &&
        @constraint(model, data.J_ineq * xv .+ data.b_ineq .>= 0)

    # CC constraints inferred from the active set.
    # Pairs where the L side is active (smaller): fix it to 0, free the R side
    if any(active_L)
        @constraint(model, data.L[active_L, :] * xv .+ data.l[active_L] .== 0)
        @constraint(model, data.R[active_L, :] * xv .+ data.r[active_L] .>= 0)
    end
    # Pairs where the R side is active (smaller): fix it to 0, free the L side
    if any(.!active_L)
        @constraint(model, data.R[.!active_L, :] * xv .+ data.r[.!active_L] .== 0)
        @constraint(model, data.L[.!active_L, :] * xv .+ data.l[.!active_L] .>= 0)
    end

    optimize!(model)
    is_solved_and_feasible(model; allow_almost = true) || return nothing

    return value.(xv)
end

function build_AL_nlp(data::MarbleData;
    λ::AbstractVector, μ::Real, ρ::Real,
    x0::AbstractVector = zeros(length(data.q)),
    s0::AbstractVector = zeros(length(data.b_ineq) + 2 * length(data.l)),
)
    n_var  = length(data.q)
    n_eq   = length(data.b_eq)
    n_ineq = length(data.b_ineq)
    n_cc   = length(data.l)
    n_s    = n_ineq + 2n_cc

    @assert length(λ)  == n_eq + n_s
    @assert length(x0) == n_var
    @assert length(s0) == n_s

    A = data.J_eq
    b = data.b_eq
    J = [data.J_ineq; data.L; data.R]
    c = [data.b_ineq; data.l; data.r]

    model = Model()
    @variable(model, x[i=1:n_var], start = x0[i])
    @variable(model, s[i=1:n_s] >= 0, start = max(s0[i], 0.0))

    r = [A * x + b; J * x + c - s]
    sL = s[n_ineq+1:n_ineq+n_cc]
    sR = s[n_ineq+n_cc+1:end]

    @objective(model, Min,
        0.5 * dot(x, data.Q, x) + dot(data.q, x)
        + μ * dot(sL, sR)
        + dot(λ, r)
        + (ρ/2) * dot(r, r)
    )
    return MathOptNLPModel(model)
end

function al_solve(data::MarbleData;
    x0           = zeros(length(data.q)),
    s0           = nothing,
    λ0           = nothing,
    μ0           = 1.0,
    ρ0           = 10.0,
    μ_max        = 1e10,
    ρ_max        = 1e10,
    feas_tol     = 1e-6,
    comp_tol     = 1e-6,
    τ            = 0.75,    # required reduction factor on residuals
    γ            = 10.0,    # penalty growth factor
    max_outer    = 50,
    sub_max_time = 30.0,
    verbose      = true,
    project      = true,
)
    n_var  = length(data.q)
    n_eq   = length(data.b_eq)
    n_ineq = length(data.b_ineq)
    n_cc   = length(data.l)
    n_s    = n_ineq + 2n_cc

    A       = data.J_eq
    b_eq    = data.b_eq
    J       = [data.J_ineq; data.L; data.R]
    c_stack = [data.b_ineq; data.l; data.r]

    x = copy(x0)
    s = s0 === nothing ? zeros(n_s)        : copy(s0)
    λ = λ0 === nothing ? zeros(n_eq + n_s) : copy(λ0)
    μ = float(μ0)
    ρ = float(ρ0)

    prev_feas, prev_comp = Inf, Inf
    sub_stats = nothing

    verbose && @printf("%4s  %10s  %10s  %10s  %9s  %9s  %10s\n",
                       "k", "feas", "comp", "subobj", "ρ", "μ", "sub_status")

    for k in 1:max_outer
        nlp = build_AL_nlp(data; λ, μ, ρ, x0 = x, s0 = s)
        sub_stats = tron(nlp; max_time = sub_max_time)

        sol = sub_stats.solution
        x   = sol[1:n_var]
        s   = sol[n_var+1:end]

        @assert all(s .>= 0) "Negative slack variable found: $s"

        r    = vcat(A * x + b_eq, J * x + c_stack - s)
        feas = norm(r, Inf)
        sL = s[n_ineq+1:n_ineq+n_cc]
        sR = s[n_ineq+n_cc+1:end]
        comp = norm((data.L * x + data.l) .* (data.R * x + data.r), Inf)

        verbose && @printf("%4d  %10.3e  %10.3e  %10.3e  %9.2e  %9.2e  %10s\n",
                           k, feas, comp, sub_stats.objective, ρ, μ, sub_stats.status)

        if project
            xproj = try_project_solution(data; x, comp_tol=1e-2)
            if !isnothing(xproj)
                return (; x = xproj, μ, ρ, status = :solved, iters = k, sub_stats)
            end
        else
            if feas ≤ feas_tol && comp ≤ comp_tol
                return (; x, s, λ, μ, ρ, feas, comp,
                          status = :solved, iters = k, sub_stats)
            end
        end

        # Multiplier update if equality residual made progress; else grow ρ.
        if feas ≤ τ * prev_feas
            λ .+= ρ .* r
        else
            ρ = min(γ * ρ, ρ_max)
        end

        # Grow complementarity penalty only when it failed to shrink.
        if comp > τ * prev_comp
            μ = min(γ * μ, μ_max)
        end

        prev_feas, prev_comp = feas, comp

        if ρ ≥ ρ_max && μ ≥ μ_max
            return (; x, s, λ, μ, ρ, feas, comp,
                      status = :max_penalty, iters = k, sub_stats)
        end
    end

    return (; x, s, λ, μ, ρ, feas = prev_feas, comp = prev_comp,
              status = :max_iter, iters = max_outer, sub_stats)
end

##

using MarbleBenchmarks

mpcc, inds = multi_robot_coordination()
data = from_NLPModel(mpcc)

result = al_solve(data; ρ0 = 100.0, μ0 = 1.0, max_outer = 100, project = true)
objective = 0.5 * dot(result.x, data.Q, result.x) + dot(data.q, result.x) + data.c0
@show objective result.status

##
name = "hs044-i"
nl_path = joinpath(@__DIR__, "..", "data", "macmpec", "ampl_nl", name * ".nl")
ampl_model = AmplModel(nl_path)
mpcc_model = mpcc_from_ampl(ampl_model)
data       = from_NLPModel(mpcc_model)

# data = MarbleData(
#     Q = I(1),
#     q = zeros(1),
#     c0 = 0.0,
#     J_eq = zeros(0, 1),
#     b_eq = zeros(0),
#     # J_ineq = zeros(0, 1),
#     # b_ineq = zeros(0),
#     J_ineq = reshape([1.0], 1, 1),
#     b_ineq = [-4.0],
#     L = reshape([-1.0], 1, 1),
#     l = [5.0],
#     R = reshape([1.0], 1, 1),
#     r = [-2.0],
# )

result = al_solve(data; ρ0 = 100.0, μ0 = 1.0, max_outer = 100, project = true)
objective = 0.5 * dot(result.x, data.Q, result.x) + dot(data.q, result.x) + data.c0
@show objective result.status