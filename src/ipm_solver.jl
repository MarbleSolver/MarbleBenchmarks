using LinearAlgebra
using SparseArrays
using QDLDL
using Printf
import ForwardDiff as FD

b(x; κ) = 0.5 * (x + √(x^2 + 4*κ))
db(x; κ) = 0.5 * (1 + x / √(x^2 + 4*κ))
ddb(x; κ) = 2*κ / (x^2 + 4*κ)^(3/2)

@Base.kwdef struct QPCCData
    Q::AbstractMatrix{Float64}
    q::AbstractVector{Float64}
    c0::Float64
    J_eq::AbstractMatrix{Float64}
    b_eq::AbstractVector{Float64}
    J_ineq::AbstractMatrix{Float64}
    b_ineq::AbstractVector{Float64}
    L::AbstractMatrix{Float64}
    l::AbstractVector{Float64}
    R::AbstractMatrix{Float64}
    r::AbstractVector{Float64}
    nvar::Int
    neq::Int
    nineq::Int
    ncc::Int
    sys_nprimals::Int
    sys_nduals::Int
    inds::Any
end

function QPCCData(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
    nvar  = length(q)
    neq   = size(J_eq, 1)
    nineq = size(J_ineq, 1)
    ncc   = size(L, 1)

    # Variable ordering in the Newton system: x, v, σ, λ, τ_L, τ_R
    off = 0
    nextblock(n) = (rng = (off + 1):(off + n); off += n; rng)
    inds = (
        x   = nextblock(nvar),
        v   = nextblock(nineq),
        σ   = nextblock(ncc),
        λ   = nextblock(neq),
        τ_L = nextblock(ncc),
        τ_R = nextblock(ncc),
    )

    return QPCCData(; Q=Q, q=q, c0=c0, J_eq=J_eq, b_eq=b_eq,
                    J_ineq=J_ineq, b_ineq=b_ineq, L=L, l=l, R=R, r=r,
                    nvar=nvar, neq=neq, nineq=nineq, ncc=ncc, inds=inds,
                    sys_nprimals = nvar + nineq + ncc,
                    sys_nduals   = neq + 2 * ncc)
end

@Base.kwdef struct QPCCSolution
    x::AbstractVector{Float64}
    v::AbstractVector{Float64}
    σ::AbstractVector{Float64}
    λ::AbstractVector{Float64}
    τ_L::AbstractVector{Float64}
    τ_R::AbstractVector{Float64}
    α::AbstractVector{Float64}
    μ::AbstractVector{Float64}
    γ_L::AbstractVector{Float64}
    γ_R::AbstractVector{Float64}
end

function QPCCSolution(data::QPCCData)
    (; nvar, nineq, ncc, neq) = data
    return QPCCSolution(
        zeros(nvar),  # x
        zeros(nineq), # v
        zeros(ncc),   # σ
        zeros(neq),   # λ
        zeros(ncc),   # τ_L
        zeros(ncc),   # τ_R
        zeros(neq),   # α
        zeros(nineq), # μ
        zeros(ncc),   # γ_L
        zeros(ncc),   # γ_R
    )
end

# ---------------------------------------------------------------------------
# KKT residual and (reduced) Newton system
# ---------------------------------------------------------------------------

# Relaxed KKT residual blocks at the current iterate (writeup §2). The barrier
# dual is recovered algebraically as μ = -b_κ(-v), so the r_μ block is
# identically zero and is dropped from the system entirely.
function kkt_residual(data::QPCCData, sol::QPCCSolution; κ, ρ)
    (; Q, q, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = data
    (; x, v, σ, λ, τ_L, τ_R, α, γ_L, γ_R) = sol

    μ = -b.(-v; κ=κ)                                      # recovered inequality dual

    r_x  = Q*x + q + J_eq'*λ + J_ineq'*μ + L'*τ_L + R'*τ_R
    r_σ  = -db.(σ; κ=κ) .* τ_L + db.(-σ; κ=κ) .* τ_R
    r_λ  = J_eq*x   + b_eq                  - 1/ρ * (λ   - α)
    r_i  = J_ineq*x + b_ineq - b.(v; κ=κ)
    r_τL = L*x + l - b.( σ; κ=κ)            - 1/ρ * (τ_L - γ_L)
    r_τR = R*x + r - b.(-σ; κ=κ)            - 1/ρ * (τ_R - γ_R)

    return (x = r_x, i = r_i, σ = r_σ, λ = r_λ, τ_L = r_τL, τ_R = r_τR)
end

# Stack the residual blocks in the Newton variable ordering (x, v, σ, λ, τ_L, τ_R).
residual_blocks(res) = vcat(res.x, res.i, res.σ, res.λ, res.τ_L, res.τ_R)

# Right-hand side of the *reduced* symmetric Newton system (writeup eq. 179).
# Eliminating Δμ folds G'(r_μ + r_i) into the x-block; since r_μ ≡ 0 this is
# simply r_x - G' r_i. The block order matches `data.inds`.
function newton_system_residual(data::QPCCData, sol::QPCCSolution; κ, ρ)
    res = kkt_residual(data, sol; κ=κ, ρ=ρ)
    return vcat(res.x - data.J_ineq' * res.i,   # x
                res.i,                           # v
                res.σ,                           # σ
                res.λ,                           # λ
                res.τ_L,                         # τ_L
                res.τ_R)                         # τ_R
end

# Matrix of the reduced symmetric Newton system (writeup eq. 179), returned as
# an upper-triangular sparse matrix for QDLDL.
function newton_system_matrix(data::QPCCData, sol::QPCCSolution; κ, ρ)
    (; Q, J_eq, J_ineq, L, R, inds, sys_nprimals, sys_nduals) = data
    (; v, σ, τ_L, τ_R) = sol
    nsys = sys_nprimals + sys_nduals

    K = spzeros(nsys, nsys)

    K[inds.x, inds.x]   = Q - J_ineq' * J_ineq
    K[inds.x, inds.v]   = J_ineq'
    K[inds.x, inds.λ]   = J_eq'
    K[inds.x, inds.τ_L] = L'
    K[inds.x, inds.τ_R] = R'

    K[inds.v, inds.v]   = Diagonal(-db.(v; κ=κ))

    K[inds.σ, inds.σ]   = Diagonal(-ddb.(σ; κ=κ) .* (τ_L + τ_R))
    K[inds.σ, inds.τ_L] = Diagonal(-db.( σ; κ=κ))
    K[inds.σ, inds.τ_R] = Diagonal( db.(-σ; κ=κ))

    K[inds.λ,   inds.λ]   = Diagonal(fill(-1/ρ, length(inds.λ)))
    K[inds.τ_L, inds.τ_L] = Diagonal(fill(-1/ρ, length(inds.τ_L)))
    K[inds.τ_R, inds.τ_R] = Diagonal(fill(-1/ρ, length(inds.τ_R)))

    return triu(Symmetric(K, :U))
end

# ---------------------------------------------------------------------------
# Iterate updates
# ---------------------------------------------------------------------------

# Apply Δ (in the reduced variable ordering) to the iterate in place, scaled by
# the line-search step size t. The AL multiplier estimates (α, γ_L, γ_R) are
# held fixed during the inner solve and are not touched here.
function apply_step!(sol::QPCCSolution, Δ::AbstractVector, t::Real, data::QPCCData)
    (; inds) = data
    sol.x   .+= t .* Δ[inds.x]
    sol.v   .+= t .* Δ[inds.v]
    sol.σ   .+= t .* Δ[inds.σ]
    sol.λ   .+= t .* Δ[inds.λ]
    sol.τ_L .+= t .* Δ[inds.τ_L]
    sol.τ_R .+= t .* Δ[inds.τ_R]
    return sol
end

# Non-mutating version used to probe candidates during the line search.
stepped(sol::QPCCSolution, Δ, t, data) = apply_step!(deepcopy(sol), Δ, t, data)

# Refresh the recovered barrier dual μ = -b_κ(-v) (only needed for output).
recover_dual!(sol::QPCCSolution; κ) = (sol.μ .= -b.(-sol.v; κ=κ); sol)

# ---------------------------------------------------------------------------
# Augmented-Lagrangian merit + line search
# ---------------------------------------------------------------------------

# Augmented Lagrangian of the relaxed problem (writeup eq. 88-96), as a scalar
# function of the *primal* variables (x, σ, v) for fixed AL estimates
# (α, γ_L, γ_R) and parameters (ρ, κ). The inequality slack d = b_κ(v) > 0 and
# its barrier dual μ = -b_κ(-v) are recovered from v; the inequality is handled
# by the log barrier (no ρ-penalty), while the equality and complementarity
# constraints get the AL multiplier + quadratic penalty treatment.
function augmented_lagrangian(data::QPCCData, sol::QPCCSolution; κ, ρ)
    (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = data
    (; x, v, σ, α, γ_L, γ_R) = sol

    d   = b.(v; κ=κ)              # inequality slacks d = b_κ(v) > 0
    μ   = -b.(-v; κ=κ)           # recovered barrier dual
    r_A = J_eq*x   + b_eq
    r_i = J_ineq*x + b_ineq - d
    r_L = L*x + l - b.( σ; κ=κ)
    r_R = R*x + r - b.(-σ; κ=κ)

    return 0.5 * dot(x, Q*x) + dot(q, x) + c0 +
           dot(μ, r_i) - κ * sum(log, d; init = 0.0) +
           dot(α,   r_A) + (ρ/2) * dot(r_A, r_A) +
           dot(γ_L, r_L) + (ρ/2) * dot(r_L, r_L) +
           dot(γ_R, r_R) + (ρ/2) * dot(r_R, r_R)
end

# Primal gradient (∇_x M, ∇_σ M, ∇_v M) of the AL merit. The merit uses the
# *implied* shifted multipliers λ̂ = α + ρ r_A, τ̂_L = γ_L + ρ r_L, τ̂_R = γ_R + ρ r_R
# (functions of the primals), not the free primal-dual variables λ, τ_L, τ_R.
# Using the retraction identity b_κ(v)b_κ(-v) = κ, the barrier/Lagrangian terms in
# ∇_v M cancel, leaving ∇_v M = b'_κ(-v) ⊙ r_i.
function al_primal_gradient(data::QPCCData, sol::QPCCSolution; κ, ρ)
    (; Q, q, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = data
    (; x, v, σ, α, γ_L, γ_R) = sol

    μ   = -b.(-v; κ=κ)
    r_A = J_eq*x   + b_eq
    r_i = J_ineq*x + b_ineq - b.(v; κ=κ)
    r_L = L*x + l - b.( σ; κ=κ)
    r_R = R*x + r - b.(-σ; κ=κ)

    λ̂  = α   + ρ * r_A
    τ̂_L = γ_L + ρ * r_L
    τ̂_R = γ_R + ρ * r_R

    g_x = Q*x + q + J_eq'*λ̂ + J_ineq'*μ + L'*τ̂_L + R'*τ̂_R
    g_σ = -db.(σ; κ=κ) .* τ̂_L + db.(-σ; κ=κ) .* τ̂_R
    g_v = db.(-v; κ=κ) .* r_i
    return g_x, g_σ, g_v
end

# Directional derivative ∇M · Δ_primal of the AL merit along a Newton step Δ.
function al_directional_derivative(data::QPCCData, sol::QPCCSolution, Δ::AbstractVector; κ, ρ)
    (; inds) = data
    g_x, g_σ, g_v = al_primal_gradient(data, sol; κ=κ, ρ=ρ)
    return dot(g_x, Δ[inds.x]) + dot(g_σ, Δ[inds.σ]) + dot(g_v, Δ[inds.v])
end

# Backtracking Armijo line search on the AL merit M. Accepts the first step size
# t satisfying the sufficient-decrease test M(w + tΔ) ≤ M(w) + c·t·(∇M·Δ).
# A non-descent direction (∇M·Δ ≥ 0) is rejected so the caller adds regularization.
function armijo_linesearch(data::QPCCData, sol::QPCCSolution, Δ::AbstractVector;
                           κ, ρ, c = 1e-4, max_backtracks = 20)
    g = al_directional_derivative(data, sol, Δ; κ=κ, ρ=ρ)
    g < 0 || return (0.0, false)                       # not a descent direction
    M0 = augmented_lagrangian(data, sol; κ=κ, ρ=ρ)
    t = 1.0
    for _ in 1:max_backtracks
        Mt = augmented_lagrangian(data, stepped(sol, Δ, t, data); κ=κ, ρ=ρ)
        Mt ≤ M0 + c * t * g && return (t, true)
        t *= 0.5
    end
    return (0.0, false)
end

# ---------------------------------------------------------------------------
# Inner Newton step with regularization (curvature) correction
# ---------------------------------------------------------------------------

const REG_SCHEDULE = [0.0; 10.0 .^ (-8.0:1.0:8.0)]

# Curvature-correction regularizer: lifts the diagonal of the "primal" blocks
# x and σ. (The slacks v and multipliers λ, τ_L, τ_R already carry their own
# -b'_κ(v) / -ρ⁻¹I diagonal, so they need no lifting.) The σ-block curvature
# -b''_κ(σ)(τ_L + τ_R) is iterate-dependent — it is positive only once ρ is
# large enough — so there is no fixed "correct" inertia signature to target.
# Instead we increase this regularization until the factorization succeeds and
# the Newton step produces sufficient residual decrease (see inner_newton_step!).
function primal_reg_block(data::QPCCData)
    d = zeros(data.sys_nprimals + data.sys_nduals)
    d[data.inds.x] .= 1.0
    d[data.inds.σ] .= 1.0
    return Diagonal(d)
end

# Factor the reduced system, increasing regularization until QDLDL succeeds on a
# nonsingular matrix. Returns the factorization and the regularization used.
function factor_regularized(data::QPCCData, K_triu; reg_schedule = REG_SCHEDULE)
    reg_block = primal_reg_block(data)
    for reg in reg_schedule
        F = try
            qdldl(K_triu + reg * reg_block)
        catch
            continue                       # singular factorization → more regularization
        end
        return F, reg
    end
    return nothing, NaN
end

# One inner iteration: a regularized (curvature-corrected) Newton direction with
# an Armijo line search on the augmented-Lagrangian merit. The regularization is
# increased until the factorization succeeds *and* the step is a descent
# direction for the AL with sufficient decrease — i.e. a non-descent direction or
# line-search failure triggers more regularization (as in Marble's C++).
function inner_newton_step!(sol::QPCCSolution, data::QPCCData; κ, ρ)
    K   = newton_system_matrix(data, sol; κ=κ, ρ=ρ)
    rhs = newton_system_residual(data, sol; κ=κ, ρ=ρ)
    reg_block = primal_reg_block(data)

    for reg in REG_SCHEDULE
        F = try
            qdldl(K + reg * reg_block)
        catch
            continue                       # singular factorization → more regularization
        end

        # if F.workspace.positive_inertia[] != data.nvar + data.ncc
        #     continue                       # wrong inertia → more regularization
        # end

        Δ = QDLDL.solve(F, -rhs)

        t, ok = armijo_linesearch(data, sol, Δ; κ=κ, ρ=ρ)
        ok || continue                     # no sufficient decrease → more regularization

        apply_step!(sol, Δ, t, data)
        return (reg = reg, step_size = t)
    end

    error("Inner Newton step failed for every regularization value")
end

# ---------------------------------------------------------------------------
# Outer (augmented-Lagrangian) updates
# ---------------------------------------------------------------------------

# First-order correction to the iterate when κ is decreased, mirroring
# `use_relax_correction` in Marble: solve K Δ = -(κ_new - κ) ∂F/∂κ and apply it.
# ∂F/∂κ is taken by forward-mode AD through the reduced residual.
function relax_correction_step!(sol::QPCCSolution, data::QPCCData; κ, κ_new, ρ)
    dκ = κ_new - κ
    ∂rhs_∂κ = FD.derivative(k -> newton_system_residual(data, sol; κ=k, ρ=ρ), κ)
    K = newton_system_matrix(data, sol; κ=κ, ρ=ρ)
    F, _ = factor_regularized(data, K)
    F === nothing && return sol            # skip the correction if we can't factor cleanly
    apply_step!(sol, QDLDL.solve(F, -(dκ .* ∂rhs_∂κ)), 1.0, data)
    return sol
end

# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

# Original problem objective ½ xᵀQx + qᵀx + c₀.
objective(data::QPCCData, sol::QPCCSolution) =
    0.5 * dot(sol.x, data.Q * sol.x) + dot(data.q, sol.x) + data.c0

# Original (unrelaxed) primal feasibility / complementarity violations (∞-norm).
function constraint_violations(data::QPCCData, sol::QPCCSolution)
    (; J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = data
    x = sol.x
    return (eq   = norm(J_eq*x + b_eq, Inf),
            ineq = norm(min.(J_ineq*x + b_ineq, 0.0), Inf),
            comp = norm((L*x + l) .* (R*x + r), Inf))
end

# Converged when the relaxed KKT residual is small *and* the original
# (unrelaxed) primal feasibility / complementarity violations are small.
# Mirrors Solver::convergence in the C++.
function is_converged(kkt_norm, viol; tol_kkt, tol_eq, tol_ineq, tol_comp)
    return kkt_norm  < tol_kkt  &&
           viol.eq   < tol_eq   &&
           viol.ineq < tol_ineq &&
           viol.comp < tol_comp
end

# ---------------------------------------------------------------------------
# Iteration logging
# ---------------------------------------------------------------------------

function print_log_header()
    @printf("%-5s  %-5s  %11s  %10s  %10s  %10s  %14s  %8s  %8s\n",
            "iter", "step", "kkt", "eq", "ineq", "comp", "obj", "kappa", "rho")
    println("─"^96)
end

function print_log_row(iter, step, kkt_norm, viol, obj, κ, ρ)
    @printf("%-5d  %-5s  %11.3e  %10.3e  %10.3e  %10.3e  %14.6f  %8.1e  %8.1e\n",
            iter, step, kkt_norm, viol.eq, viol.ineq, viol.comp, obj, κ, ρ)
end

# ---------------------------------------------------------------------------
# Main solve
# ---------------------------------------------------------------------------

"""
    solve_ipm(data::QPCCData; kwargs...)

Augmented-Lagrangian interior-point solver for the relaxed QPCC (writeup §3),
mirroring the Marble C++ algorithm conceptually:

  * INNER loop — while the relaxed KKT residual is large, take regularized
    (inertia-corrected) Newton steps with an Armijo line search on ½‖F‖².
  * OUTER step — once the residual is small, shift the AL multiplier estimates
    (α ← λ, γ_L ← τ_L, γ_R ← τ_R) and update the parameters: first drive the
    penalty ρ up to `ρ_max`, then crank the relaxation κ down toward `κ_min`
    (optionally with a first-order correction to the iterate).
"""
function solve_ipm(data::QPCCData;
                   max_iters            = 1000,
                   κ0 = 1e-1, κ_min = 1e-7, κ_scaling = 0.5,
                   ρ0 = 1e1,  ρ_max = 1e6, ρ_scaling = 10.0,
                   outer_step_kkt_norm  = 1e-6,
                   tol_kkt = 1e-4, tol_eq = 1e-4, tol_ineq = 1e-4, tol_comp = 1e-5,
                   use_relax_correction = false,
                   verbose              = true)

    sol = QPCCSolution(data)
    κ, ρ = κ0, ρ0
    β = 1.0                        # shrinks the inner tolerance near the end
    iters_inner = iters_outer = 0
    last_outer_iter = -1

    verbose && print_log_header()

    for iter in 1:max_iters
        kkt_norm = norm(residual_blocks(kkt_residual(data, sol; κ=κ, ρ=ρ)), Inf)
        viol = constraint_violations(data, sol)
        obj  = objective(data, sol)

        if is_converged(kkt_norm, viol; tol_kkt, tol_eq, tol_ineq, tol_comp)
            verbose && print_log_row(iter, "conv", kkt_norm, viol, obj, κ, ρ)
            recover_dual!(sol; κ=κ)
            return (; sol, converged = true, iters = iter, iters_inner, iters_outer)
        end

        is_outer = kkt_norm < β * outer_step_kkt_norm
        verbose && print_log_row(iter, is_outer ? "outer" : "inner", kkt_norm, viol, obj, κ, ρ)

        if is_outer
            # ----- OUTER STEP: AL multiplier + parameter update -----
            iters_outer += 1

            # Once κ can no longer shrink, tighten the inner tolerance instead.
            if last_outer_iter == iter - 1 && κ ≤ κ_min
                β *= 0.1
            end

            # Shift AL multiplier estimates to the current shifted multipliers.
            sol.α   .= sol.λ
            sol.γ_L .= sol.τ_L
            sol.γ_R .= sol.τ_R

            if ρ < ρ_max
                ρ = min(ρ * ρ_scaling, ρ_max)            # first: max out the penalty
            else
                κ_new = max(κ * κ_scaling, κ_min)         # then: crank κ down
                if use_relax_correction && κ_new != κ
                    relax_correction_step!(sol, data; κ=κ, κ_new=κ_new, ρ=ρ)
                end
                κ = κ_new
            end

            last_outer_iter = iter
        else
            # ----- INNER STEP: regularized Newton + Armijo line search -----
            iters_inner += 1
            inner_newton_step!(sol, data; κ=κ, ρ=ρ)
        end
    end

    recover_dual!(sol; κ=κ)
    verbose && @warn "solve_ipm hit max_iters without converging" max_iters
    return (; sol, converged = false, iters = max_iters, iters_inner, iters_outer)
end

## Test

using MarbleBenchmarks, Marble

name = Symbol("bard1")
mpcc = get_problem(name)
data = QPCCData(mpcc_to_marble(mpcc)...)

result = solve_ipm(
    data; verbose=true,
    κ_scaling=0.5
)