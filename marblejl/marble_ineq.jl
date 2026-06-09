using LinearAlgebra
using Marble, MarbleBenchmarks
import ForwardDiff as FD

# r = sqrt(t^2 + 4κ), computed with hypot to avoid overflow/underflow of t^2.
@inline _rκ(t, κ) = hypot(t, 2 * sqrt(κ))

@inline function bκ(t; κ)
    # bκ(t) = (t + r) / 2. For t < 0 this cancels catastrophically (bκ -> 0+),
    # so use the rationalized form bκ(t) = 2κ / (r - t), which adds positives.
    r = _rκ(t, κ)
    return t ≥ 0 ? 0.5 * (t + r) : 2κ / (r - t)
end

@inline function dbκ(t; κ)
    # bκ'(t) = 0.5 * (1 + t/r) = bκ(t) / r; ratio of two positive stable values.
    return bκ(t; κ) / _rκ(t, κ)
end

@inline function ddbκ(t; κ)
    # bκ''(t) = 2κ / (t^2 + 4κ)^{3/2} = 2κ / r^3.
    r = _rκ(t, κ)
    return 2κ / r^3
end

@inline function inv_bκ(y; κ)
    y > 0 || throw(ArgumentError("bκ inverse requires y > 0, got $y"))
    return y - κ / y
end

function make_data(Q, q, c0, A, b, G, h, L, l, R, r)
    nvar = length(q)
    neq = size(A, 1)
    nineq = size(G, 1)
    ncc = size(L, 1)

    @assert size(Q) == (nvar, nvar)
    @assert size(A, 2) == nvar
    @assert size(G, 2) == nvar
    @assert size(L, 2) == nvar
    @assert size(R) == size(L)

    @assert length(b) == neq
    @assert length(h) == nineq
    @assert length(l) == ncc
    @assert length(r) == ncc

    inds = build_indices(nvar, neq, nineq, ncc)

    return (;
        Q, q, c0,
        A, b,
        G, h,
        L, l,
        R, r,
        nvar, neq, nineq, ncc,
        inds
    )
end

function split_vars(sol, data)
    I = data.inds
    return (; [k => sol[getfield(I, k)] for k in keys(I)]...)
end

function block_ranges(sizes)
    offsets = cumsum([0; sizes])
    return [offsets[i] + 1 : offsets[i + 1] for i in eachindex(sizes)]
end

function build_indices(nvar, neq, nineq, ncc)
    full = block_ranges([nvar, nineq, ncc, ncc, ncc, neq, nineq, ncc, ncc])

    inds = (;
        x = full[1],
        v = full[2],
        s = full[3],
        t = full[4],
        u = full[5],
        λ = full[6],
        μ = full[7],
        nu = full[8],
        eta = full[9],
    )

    return inds
end

function choose_branch(data, x, i)
    Lval = dot(view(data.L, i, :), x) + data.l[i]
    Rval = dot(view(data.R, i, :), x) + data.r[i]

    if Lval < 0 ≤ Rval
        return :R
    elseif Rval < 0 ≤ Lval
        return :L
    elseif Rval > Lval
        return :R
    elseif Lval > Rval
        return :L
    else
        return :R
    end
end

function branch_equalities(data, sides)
    rows = Vector{Vector{Float64}}()
    rhs = Float64[]

    for i in 1:data.neq
        push!(rows, collect(view(data.A, i, :)))
        push!(rhs, -data.b[i])
    end

    for i in 1:data.ncc
        if sides[i] == :L
            push!(rows, collect(view(data.R, i, :)))
            push!(rhs, -data.r[i])
        else
            push!(rows, collect(view(data.L, i, :)))
            push!(rhs, -data.l[i])
        end
    end

    if isempty(rows)
        return zeros(0, data.nvar), Float64[]
    end

    K = reduce(vcat, transpose.(rows))
    return K, rhs
end

function project_equalities_nearest(K, rhs, xref)
    isempty(rhs) && return copy(xref)
    return xref + pinv(K) * (rhs - K * xref)
end

function branch_projection(data, sides; xref = zeros(data.nvar), enforce_nonnegative = true)
    K, rhs = branch_equalities(data, sides)
    x = project_equalities_nearest(K, rhs, xref)

    enforce_nonnegative || return x

    # The equality projection alone can pick the wrong point in an underdetermined
    # branch manifold. Add the most violated nonnegativity rows as equalities to
    # get a finite, domain-friendly trial near the user's/model's start.
    H = [data.G; data.L; data.R]
    h = [data.h; data.l; data.r]
    active = Int[]

    for _ in 1:(size(H, 1) + 1)
        isempty(h) && return x
        vals = H * x + h
        idx = argmin(vals)
        vals[idx] ≥ -1e-9 && return x
        idx in active && return x

        push!(active, idx)
        K = [K; transpose(H[idx, :])]
        push!(rhs, -h[idx])
        x = project_equalities_nearest(K, rhs, xref)
    end

    return x
end

# Initial point that is (1) strictly feasible for the relaxation barrier
# (σ = -s-t > 0), and (2) branch-biased (one of s,t positive, the other
# negative). The second part matters: a fully symmetric or both-negative start
# on a swap-symmetric MPCC is drawn toward the biactive relaxation and can miss
# the asymmetric complementarity solution entirely.
function initialize_lifted_coordinates!(sol, data; κ = 1e-1, σ0 = 1.0, branch0 = 1.0)
    branch = max(branch0, sqrt(κ))
    slack_floor = min(sqrt(κ), 0.1 * κ / branch)
    sol[data.inds.v] .= inv_bκ.(max.(data.G * sol[data.inds.x] .+ data.h, slack_floor); κ = κ)

    for i in 1:data.ncc
        Lval = dot(view(data.L, i, :), sol[data.inds.x]) + data.l[i]
        Rval = dot(view(data.R, i, :), sol[data.inds.x]) + data.r[i]
        side = choose_branch(data, sol[data.inds.x], i)

        if side == :L
            yL = max(Lval, branch)
            yR = min(max(Rval, slack_floor), 0.1 * κ / yL)
        else
            yR = max(Rval, branch)
            yL = min(max(Lval, slack_floor), 0.1 * κ / yR)
        end

        sol[data.inds.s[i]] = inv_bκ(yL; κ = κ)
        sol[data.inds.t[i]] = inv_bκ(yR; κ = κ)
    end

    σ = -sol[data.inds.s] .- sol[data.inds.t]
    sol[data.inds.u] .= inv_bκ.(max.(σ, max(σ0, sqrt(κ))); κ = κ)

    return sol
end

function init_sol(data; κ = 1e-1, σ0 = 1.0, branch0 = 1.0, x0 = nothing)
    sol = zeros(sum(length, values(data.inds)))

    if x0 === nothing
        sides = [choose_branch(data, sol[data.inds.x], i) for i in 1:data.ncc]
        sol[data.inds.x] .= branch_projection(data, sides)
    else
        sides = [choose_branch(data, x0, i) for i in 1:data.ncc]
        sol[data.inds.x] .= branch_projection(data, sides; xref = x0)
    end

    initialize_lifted_coordinates!(sol, data; κ = κ, σ0 = σ0, branch0 = branch0)
    return sol
end

function reset_shifted_multipliers!(sol, duals, data; κ, ρ)
    (; x, v, s, t) = split_vars(sol, data)
    (; λ_est, μ_est, nu_est, eta_est) = duals
    (; A, b, G, h, L, l, R, r) = data

    sol[data.inds.λ] .= λ_est .+ ρ .* (A * x .+ b)
    sol[data.inds.μ] .= μ_est .+ ρ .* (G * x .+ h .- bκ.(v; κ = κ))
    sol[data.inds.nu] .= nu_est .+ ρ .* (L * x .+ l .- bκ.(s; κ = κ))
    sol[data.inds.eta] .= eta_est .+ ρ .* (R * x .+ r .- bκ.(t; κ = κ))

    return sol
end

function prepare_initial_sol(sol, duals, data; κ, ρ, branch0 = 1.0)
    sol = copy(sol)
    ncc = data.ncc

    if ncc > 0
        σ = -sol[data.inds.s] .- sol[data.inds.t]
        bad_domain = .!isfinite.(σ) .| (σ .<= 0)
        symmetric = abs.(sol[data.inds.s] .- sol[data.inds.t]) .<=
                    10 * eps(real(eltype(sol))) .* max.(1, abs.(σ))

        if any(bad_domain) || any(symmetric)
            if norm(sol[data.inds.x], Inf) ≤ 10 * eps(real(eltype(sol)))
                sol .= init_sol(data; κ = κ, branch0 = branch0)
                reset_shifted_multipliers!(sol, duals, data; κ = κ, ρ = ρ)
                return sol
            end

            initialize_lifted_coordinates!(sol, data; κ = κ, branch0 = branch0)
            reset_shifted_multipliers!(sol, duals, data; κ = κ, ρ = ρ)
            return sol
        end
    end

    reset_shifted_multipliers!(sol, duals, data; κ = κ, ρ = ρ)
    return sol
end

function Pmat(data; ρ)
    return Diagonal([
        ones(data.nvar);
        ones(data.nineq);
        ones(data.ncc);
        ones(data.ncc);
        ones(data.ncc);
        1/ρ * ones(data.neq);
        1/ρ * ones(data.nineq);
        1/ρ * ones(data.ncc);
        1/ρ * ones(data.ncc);
    ])
end

function Rmat(data, v, u; κ)
    return Diagonal([
        ones(data.nvar);
        -dbκ.(v; κ = κ);
        -ones(data.ncc);
        -ones(data.ncc);
        -dbκ.(-u; κ = κ);
        -ones(data.neq);
        -ones(data.nineq);
        -ones(data.ncc);
        -ones(data.ncc);
    ])
end

function Smat(data, v, u; κ)
    return Diagonal([
        ones(data.nvar);
        dbκ.(v; κ = κ).^(-1/2);
        ones(data.ncc);
        ones(data.ncc);
        dbκ.(-u; κ = κ).^(-1/2);
        ones(data.neq);
        ones(data.nineq);
        ones(data.ncc);
        ones(data.ncc);
    ])
end

function kkt_residual(sol, duals, data; κ, ρ)
    (; x, v, s, t, u, λ, μ, nu, eta) = split_vars(sol, data)
    (; λ_est, μ_est, nu_est, eta_est) = duals
    (; Q, q, A, b, G, h, L, l, R, r) = data

    return [
        # g_x
        Q * x + q + A' * λ + G' * μ + L' * nu + R' * eta;
        # g_v
        μ + bκ.(-v; κ = κ);
        # g_s
        dbκ.(s; κ = κ) .* nu - bκ.(-u; κ = κ);
        # g_t
        dbκ.(t; κ = κ) .* eta - bκ.(-u; κ = κ);
        # g_u
        s + t + bκ.(u; κ = κ);
        # c_λ
        1/ρ * (λ - λ_est) - (A * x + b);
        # c_μ
        1/ρ * (μ - μ_est) - (G * x + h - bκ.(v; κ = κ));
        # c_nu
        1/ρ * (nu - nu_est) - (L * x + l - bκ.(s; κ = κ));
        # c_eta
        1/ρ * (eta - eta_est) - (R * x + r - bκ.(t; κ = κ));
    ]
end

function build_kkt_system(sol, duals, data; κ, ρ)
    (; x, v, s, t, u, λ, μ, nu, eta) = split_vars(sol, data)

    R = Rmat(data, v, u; κ = κ)
    S = Smat(data, v, u; κ = κ)

    J_full = FD.jacobian(z -> kkt_residual(z, duals, data; κ = κ, ρ = ρ), sol)
    J_bal = S * R * J_full * S

    # M = Diagonal([
    #     ones(data.nvar + data.nineq + 3 * data.ncc);
    #     zeros(data.neq + data.nineq + 2 * data.ncc);
    # ])
    M = zeros(size(J_full))

    ϵ = 1e-6

    return (; J_full=J_full + ϵ*M, J_bal=J_bal + ϵ*M, R, S)
end

function Δ_step(sol, duals, data; κ, ρ, dbg = false)
    r = kkt_residual(sol, duals, data; κ = κ, ρ = ρ)
    (; J_full, J_bal, R, S) = build_kkt_system(sol, duals, data; κ = κ, ρ = ρ)

    dbg && @info "cond" J_full = cond(J_full) J_bal = cond(J_bal)

    # Solve the balanced symmetric system, then map back: Δ = S Δ_bal.
    Δ_bal = -J_bal \ (S * R * r)
    Δ_full = S * Δ_bal

    return Δ_full
end

function augmented_lagrangian(sol, duals, data; κ, ρ)
    # Scalar AL merit for the inner minimization, as a function of the primal
    # coordinates (x, v, s, t) only; the multiplier *estimates* in `duals` are
    # held fixed. The shifted multipliers and u in `sol` do not enter
    (; x, v, s, t) = split_vars(sol, data)
    (; λ_est, μ_est, nu_est, eta_est) = duals
    (; Q, q, c0, A, b, G, h, L, l, R, r) = data

    T = eltype(x)

    # Implicit slacks: w = b_κ(v) > 0 (ordinary inequality) and σ = -s-t > 0
    # (relaxation). The barriers are -κ Σ log(w) and -κ Σ log(σ); w > 0 always,
    # but σ can leave the domain, so return +∞ there for the line search to reject.
    σ = -s .- t
    any(≤(zero(T)), σ) && return convert(T, Inf)

    w = bκ.(v; κ = κ)

    # Primal constraint residuals.
    rA = A * x .+ b
    rG = G * x .+ h .- w
    rL = L * x .+ l .- bκ.(s; κ = κ)
    rR = R * x .+ r .- bκ.(t; κ = κ)

    # Objective + log-barriers (init keeps empty barrier blocks well-defined,
    # e.g. nineq = 0 ⇒ no w-barrier, ncc = 0 ⇒ no σ-barrier).
    val = 0.5 * dot(x, Q * x) + dot(q, x) + c0
    val -= κ * sum(log, w; init = zero(T))
    val -= κ * sum(log, σ; init = zero(T))

    # Augmented-Lagrangian multiplier + penalty terms.
    val += dot(λ_est, rA)  + (ρ / 2) * dot(rA, rA)
    val += dot(μ_est, rG)  + (ρ / 2) * dot(rG, rG)
    val += dot(nu_est, rL) + (ρ / 2) * dot(rL, rL)
    val += dot(eta_est, rR) + (ρ / 2) * dot(rR, rR)

    return val
end

function fraction_to_boundary(sol, Δ, data; τ = 0.995)
    (; s, t) = split_vars(sol, data)
    Δs = Δ[data.inds.s]
    Δt = Δ[data.inds.t]

    σ = -s .- t
    Δσ = -Δs .- Δt
    α = 1.0

    for i in eachindex(σ)
        if Δσ[i] < 0
            α = min(α, τ * σ[i] / (-Δσ[i]))
        end
    end

    return max(0.0, min(1.0, α))
end

# Inner line search on the KKT residual norm ½‖F‖². The lifted Newton step solves
# J Δ = -F exactly, so ∇(½‖F‖²)·Δ = Fᵀ J Δ = -‖F‖² < 0 for a well-solved Newton
# system. The fraction-to-boundary cap keeps σ = -s-t strictly positive.
function linesearch(sol, duals, data, Δ; κ, ρ, β=0.5, c1=1e-4, αmin=1e-12)
    function merit(z)
        (; s, t) = split_vars(z, data)
        any(≤(zero(eltype(z))), -s .- t) && return convert(eltype(z), Inf)  # keep σ = -s-t > 0
        return 0.5 * sum(abs2, kkt_residual(z, duals, data; κ = κ, ρ = ρ))
    end

    ϕ0  = merit(sol)
    if !isfinite(ϕ0)
        @warn "linesearch called from outside the finite domain"
        return 0.0, false
    end

    dϕ0 = -2 * ϕ0                         # = -‖F‖² (descent for the Newton direction)

    α = fraction_to_boundary(sol, Δ, data)
    while α > αmin && merit(sol + α * Δ) > ϕ0 + c1 * α * dϕ0
        α *= β
    end

    if α ≤ αmin
        @warn "linesearch failed" α
        return 0.0, false
    end

    return α, true
end

# Constraint violations / complementarity / stationarity at the current iterate.
function measures(sol, duals, data; κ, ρ)
    (; x, v, s, t) = split_vars(sol, data)
    (; A, b, G, h, L, l, R, r) = data

    rA = A * x .+ b
    rG = G * x .+ h .- bκ.(v; κ = κ)
    rL = L * x .+ l .- bκ.(s; κ = κ)
    rR = R * x .+ r .- bκ.(t; κ = κ)
    feas = max(norm(rA, Inf), norm(rG, Inf), norm(rL, Inf), norm(rR, Inf))

    comp = norm((L * x .+ l) .* (R * x .+ r), Inf)   # complementarity gap (≈ κ)

    F = kkt_residual(sol, duals, data; κ = κ, ρ = ρ)
    inner = norm(F, Inf)

    return (; feas, comp, inner)
end

# Dual update: at inner convergence c_• ≈ 0, so the shifted-multiplier blocks of
# `sol` already equal the AL update λ_est + ρ r_•. Accepting the duals is a copy.
function accept_duals(sol, data)
    (; λ, μ, nu, eta) = split_vars(sol, data)
    return (; λ_est = λ, μ_est = μ, nu_est = nu, eta_est = eta)
end

# Inner Newton + linesearch for fixed (κ, ρ, duals), to ‖F‖∞ ≤ ω.
function inner_solve(sol, duals, data; κ, ρ, ω = 1e-8, maxiter = 50)
    for i in 1:maxiter
        F = kkt_residual(sol, duals, data; κ = κ, ρ = ρ)
        norm(F, Inf) ≤ ω && return sol, i, true

        Δ = Δ_step(sol, duals, data; κ = κ, ρ = ρ)
        α, accepted = linesearch(sol, duals, data, Δ; κ = κ, ρ = ρ)
        accepted || return sol, i, false
        sol = sol .+ α .* Δ
    end
    return sol, maxiter, false
end

# Outer augmented-Lagrangian / homotopy loop.
#
# For fixed κ, first solve the relaxed equality system reasonably well. Only then
# advance the homotopy κ ← σκ. If the inner solve succeeds but relaxed feasibility
# is still loose, keep κ fixed and strengthen ρ; otherwise κ gets cranked down
# before the current relaxed problem has actually been solved.
function solve(sol, duals, data;
               κ = 1e-1, ρ = 1e1,
               κ_min = 1e-10, ρ_max = 1e12,
               γ = 7.0, σ = 0.2, θ = 0.25,
               ε_feas = 1e-7, ε_comp = 1e-7,
               κ_update_feas_factor = 0.1,
               κ_update_comp_factor = 1.5,
               ρ_init_factor = 5e2,
               maxouter = 50, verbose = true)
    ρ = max(ρ, ρ_init_factor * max(1.0, norm(data.q, Inf), norm(data.Q, Inf)))
    sol = prepare_initial_sol(sol, duals, data; κ = κ, ρ = ρ)
    prev_feas = Inf

    for k in 1:maxouter
        ω = max(1e-11, 1e-2 * κ)                       # don't over-solve at large κ
        sol, iters, ok = inner_solve(sol, duals, data; κ = κ, ρ = ρ, ω = ω)
        m = measures(sol, duals, data; κ = κ, ρ = ρ)

        κ_feas_tol = max(ε_feas, κ_update_feas_factor * κ)
        κ_comp_tol = max(ε_comp, κ_update_comp_factor * κ)
        relaxed_solved = (m.feas ≤ κ_feas_tol) && (m.comp ≤ κ_comp_tol)

        verbose && @info "outer $k" κ ρ feas=m.feas comp=m.comp κ_feas_tol κ_comp_tol iters ok

        # Converged on the actual targets — no need to chase κ → κ_min.
        if m.feas ≤ ε_feas && m.comp ≤ ε_comp
            return (; sol, duals, κ, ρ, converged = true, k, inner_failed = false)
        end

        if !ok && !relaxed_solved
            return (; sol, duals, κ, ρ, converged = false, k, inner_failed = true)
        end

        duals = accept_duals(sol, data)                 # dual update (always)

        if relaxed_solved
            κ_next = max(κ_min, σ * κ)
            if κ_next < κ
                κ = κ_next
                prev_feas = Inf
                continue
            end
        end

        # Keep κ fixed until the current relaxed problem is tight enough. If the
        # relaxed feasibility is loose or stalled, strengthen the AL penalty.
        if !relaxed_solved || m.feas > θ * prev_feas
            ρ = min(ρ_max, γ * ρ)
        end

        prev_feas = m.feas
    end

    return (; sol, duals, κ, ρ, converged = false, k = maxouter, inner_failed = false)
end

function macmpec(name)
    problem = get_problem(name)
    Q, q, c0, A, b, G, h, L, l, R, r = jump_to_marble(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)
    data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)

    sol = init_sol(data; x0 = problem.model.meta.x0)
    duals = (
        λ_est = zeros(data.neq),
        μ_est = zeros(data.nineq),
        nu_est = zeros(data.ncc),
        eta_est = zeros(data.ncc)
    )

    return data, sol, duals
end

if abspath(PROGRAM_FILE) == @__FILE__
    data, sol, duals = macmpec(Symbol("bilevel2"))
    res = solve(sol, duals, data; κ = 0.1, ρ = 10.0)
    x = res.sol[data.inds.x]
    obj = 0.5 * dot(x, data.Q * x) + dot(data.q, x) + data.c0

    @show res.converged res.k res.κ res.ρ obj
    @show x
end
