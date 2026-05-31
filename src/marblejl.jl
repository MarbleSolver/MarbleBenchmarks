using LinearAlgebra
using Printf
using Marble: MarbleData

"""
    JuliaMarbleOptions(; kwargs...)

Options for the experimental pure-Julia Marble solver in this file. The solver
uses a log-domain primal-dual Newton method for the inequalities of each fixed
`kappa` relaxation, then reduces `kappa`.
"""
Base.@kwdef mutable struct JuliaMarbleOptions
    kappa_initial::Float64 = 1e-1
    kappa_min::Float64 = 1e-8
    kappa_scaling::Float64 = 0.1

    mu_initial::Float64 = 1e-4
    mu_min::Float64 = 1e-10
    mu_scaling::Float64 = 0.1

    max_outer_iters::Int = 20
    max_mu_iters::Int = 20
    max_newton_iters::Int = 60
    max_linesearch_iters::Int = 24

    newton_tolerance::Float64 = 1e-8
    convergence_eq_violation::Float64 = 1e-8
    convergence_ineq_violation::Float64 = 1e-8
    convergence_comp_violation::Float64 = 1e-8

    regularization_initial::Float64 = 1e-10
    regularization_scaling::Float64 = 10.0
    regularization_max::Float64 = 1e8

    armijo::Float64 = 1e-4
    min_step::Float64 = 1e-10
    verbosity::Int = 0
end

Base.@kwdef struct JuliaMarbleResult
    x::Vector{Float64}
    status::Symbol
    objective::Float64
    kappa::Float64
    mu::Float64
    iterations_outer::Int
    iterations_mu::Int
    iterations_newton::Int
    residual_eq::Float64
    residual_ineq::Float64
    residual_comp::Float64
    residual_endpoint::Float64
    kkt_residual::Float64
    solve_time_s::Float64
end

julia_marble_converged(result::JuliaMarbleResult) = result.status == :converged
julia_marble_solution(result::JuliaMarbleResult) = result.x
julia_marble_solve_time_s(result::JuliaMarbleResult) = result.solve_time_s

_as_dense(A) = Matrix{Float64}(A)
_as_vector(v) = Vector{Float64}(v)
_safe_exp(x) = exp(clamp(x, -60.0, 60.0))
_safe_exp(v::AbstractVector) = _safe_exp.(v)
_inf_norm(v::AbstractVector) = isempty(v) ? 0.0 : norm(v, Inf)

function _check_data_shapes(data::MarbleData)
    n = length(data.q)
    checks = (
        size(data.Q) == (n, n),
        size(data.J_eq, 2) == n,
        size(data.J_ineq, 2) == n,
        size(data.L, 2) == n,
        size(data.R, 2) == n,
        length(data.b_eq) == size(data.J_eq, 1),
        length(data.b_ineq) == size(data.J_ineq, 1),
        length(data.l) == size(data.L, 1),
        length(data.r) == size(data.R, 1),
        size(data.L, 1) == size(data.R, 1),
    )
    all(checks) || throw(ArgumentError("Inconsistent MarbleData dimensions"))
    arrays = (data.Q, data.q, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq,
              data.L, data.l, data.R, data.r)
    all(A -> all(isfinite, A), arrays) || throw(ArgumentError("MarbleData contains non-finite values"))
    return true
end

function _initial_point(data::MarbleData, kappa::Float64, x0)
    n = length(data.q)
    m = length(data.l)

    if isnothing(x0)
        target = max(sqrt(max(kappa, eps(Float64))) * 0.5, 1e-6)
        A0 = [Matrix{Float64}(data.J_eq); Matrix{Float64}(data.L); Matrix{Float64}(data.R)]
        b0 = vcat(-Vector{Float64}(data.b_eq),
                  fill(target, m) .- Vector{Float64}(data.l),
                  fill(target, m) .- Vector{Float64}(data.r))
        x = isempty(b0) ? zeros(Float64, n) : Vector{Float64}(A0 \ b0)
    else
        x = Vector{Float64}(x0[1:n])
    end
    s = zeros(Float64, m)
    t = zeros(Float64, m)

    if m > 0
        a = _as_vector(data.L * x + data.l)
        b = _as_vector(data.R * x + data.r)
        floor_val = max(sqrt(max(kappa, eps(Float64))) * 0.5, 1e-8)
        s .= log.(max.(a, floor_val))
        t .= log.(max.(b, floor_val))

        log_kappa = log(max(kappa, eps(Float64)))
        for i in 1:m
            excess = s[i] + t[i] - (log_kappa - 1.0)
            if excess > 0
                s[i] -= 0.5 * excess
                t[i] -= 0.5 * excess
            end
        end
    end

    return vcat(x, s, t)
end

function _split_z(y::AbstractVector, n::Int, m::Int)
    x = view(y, 1:n)
    s = view(y, n + 1:n + m)
    t = view(y, n + m + 1:n + 2m)
    return x, s, t
end

function _residual_data(data::MarbleData, y::AbstractVector, lambda::AbstractVector,
                        v::AbstractVector, mu::Float64, kappa::Float64)
    Q = _as_dense(data.Q)
    q = _as_vector(data.q)
    J_eq = _as_dense(data.J_eq)
    b_eq = _as_vector(data.b_eq)
    J_ineq = _as_dense(data.J_ineq)
    b_ineq = _as_vector(data.b_ineq)
    L = _as_dense(data.L)
    l = _as_vector(data.l)
    R = _as_dense(data.R)
    r = _as_vector(data.r)

    n = length(q)
    m = length(l)
    p0 = length(b_eq)
    mi = length(b_ineq)
    N = n + 2m

    x, s, t = _split_z(y, n, m)
    es = _safe_exp(s)
    et = _safe_exp(t)

    c = Vector{Float64}(undef, p0 + 2m)
    p0 > 0 && (c[1:p0] .= J_eq * x + b_eq)
    if m > 0
        c[p0 + 1:p0 + m] .= L * x + l .- es
        c[p0 + m + 1:p0 + 2m] .= R * x + r .- et
    end

    h = Vector{Float64}(undef, mi + m)
    mi > 0 && (h[1:mi] .= J_ineq * x + b_ineq)
    m > 0 && (h[mi + 1:mi + m] .= log(max(kappa, eps(Float64))) .- s .- t)

    Jc = zeros(Float64, p0 + 2m, N)
    p0 > 0 && (Jc[1:p0, 1:n] .= J_eq)
    if m > 0
        Jc[p0 + 1:p0 + m, 1:n] .= L
        Jc[p0 + m + 1:p0 + 2m, 1:n] .= R
        for i in 1:m
            Jc[p0 + i, n + i] = -es[i]
            Jc[p0 + m + i, n + m + i] = -et[i]
        end
    end

    Jh = zeros(Float64, mi + m, N)
    mi > 0 && (Jh[1:mi, 1:n] .= J_ineq)
    if m > 0
        for i in 1:m
            Jh[mi + i, n + i] = -1.0
            Jh[mi + i, n + m + i] = -1.0
        end
    end

    sqrt_mu = sqrt(mu)
    nu = sqrt_mu .* _safe_exp(v)
    target_slack = sqrt_mu .* _safe_exp(-v)

    grad = zeros(Float64, N)
    grad[1:n] .= Q * x + q

    H = zeros(Float64, N, N)
    H[1:n, 1:n] .= Q
    if m > 0
        lambda_L = view(lambda, p0 + 1:p0 + m)
        lambda_R = view(lambda, p0 + m + 1:p0 + 2m)
        for i in 1:m
            H[n + i, n + i] += -lambda_L[i] * es[i]
            H[n + m + i, n + m + i] += -lambda_R[i] * et[i]
        end
    end

    r_stat = grad + Jc' * lambda - Jh' * nu
    r_slack = h - target_slack
    residual = vcat(r_stat, c, r_slack)

    return (; residual, r_stat, c, h, Jc, Jh, H, nu, target_slack)
end

function _merit(data::MarbleData, y, lambda, v, mu, kappa)
    rd = _residual_data(data, y, lambda, v, mu, kappa)
    phi = 0.5 * dot(rd.residual, rd.residual)
    return isfinite(phi) ? phi : Inf
end

function _regularized_newton_step(rd, opts::JuliaMarbleOptions)
    N = size(rd.H, 1)
    p = size(rd.Jc, 1)
    mineq = size(rd.Jh, 1)

    K = zeros(Float64, N + p + mineq, N + p + mineq)
    K[1:N, 1:N] .= rd.H
    p > 0 && (K[1:N, N + 1:N + p] .= rd.Jc')
    p > 0 && (K[N + 1:N + p, 1:N] .= rd.Jc)
    if mineq > 0
        K[1:N, N + p + 1:end] .= -rd.Jh' * Diagonal(rd.nu)
        K[N + p + 1:end, 1:N] .= rd.Jh
        K[N + p + 1:end, N + p + 1:end] .= Diagonal(rd.target_slack)
    end

    rhs = -rd.residual
    reg = 0.0
    while reg <= opts.regularization_max
        Kreg = reg == 0.0 ? K : K + reg * I
        step = try
            Kreg \ rhs
        catch
            nothing
        end

        if step !== nothing && all(isfinite, step)
            return step, reg
        end
        reg = reg == 0.0 ? opts.regularization_initial : reg * opts.regularization_scaling
    end

    error("Failed to solve regularized Newton system")
end

function _solve_fixed_kappa!(data::MarbleData, y::Vector{Float64}, lambda::Vector{Float64},
                             v::Vector{Float64}, kappa::Float64, opts::JuliaMarbleOptions)
    n_newton = 0
    n_mu = 0
    last_residual = Inf
    mu = opts.mu_initial

    while mu >= opts.mu_min && n_mu < opts.max_mu_iters
        n_mu += 1
        for _ in 1:opts.max_newton_iters
            n_newton += 1
            rd = _residual_data(data, y, lambda, v, mu, kappa)
            res_norm = norm(rd.residual)
            last_residual = res_norm
            res_norm <= opts.newton_tolerance && break

            step, reg = _regularized_newton_step(rd, opts)
            N = length(y)
            p = length(lambda)

            dy = view(step, 1:N)
            dlambda = view(step, N + 1:N + p)
            dv = view(step, N + p + 1:length(step))

            phi0 = 0.5 * dot(rd.residual, rd.residual)
            alpha = 1.0
            accepted = false
            for _ls in 1:opts.max_linesearch_iters
                y_trial = y + alpha .* dy
                lambda_trial = lambda + alpha .* dlambda
                v_trial = v + alpha .* dv
                phi_trial = _merit(data, y_trial, lambda_trial, v_trial, mu, kappa)
                if phi_trial <= (1.0 - opts.armijo * alpha) * phi0 || phi_trial < phi0
                    y .= y_trial
                    lambda .= lambda_trial
                    v .= v_trial
                    accepted = true
                    break
                end
                alpha *= 0.5
                alpha < opts.min_step && break
            end

            if !accepted
                y .+= opts.min_step .* dy
                lambda .+= opts.min_step .* dlambda
                v .+= opts.min_step .* dv
            end

            if opts.verbosity >= 3
                @printf("    mu %.3e residual %.3e step %.3e reg %.3e\n",
                        mu, res_norm, accepted ? alpha : opts.min_step, reg)
            end
        end
        mu *= opts.mu_scaling
    end

    mu_used = max(mu / opts.mu_scaling, opts.mu_min)
    last_residual = norm(_residual_data(data, y, lambda, v, mu_used, kappa).residual)
    return n_mu, n_newton, last_residual, mu_used
end

function _residual_summary(data::MarbleData, x::AbstractVector, kkt_residual::Float64)
    a = _as_vector(data.L * x + data.l)
    b = _as_vector(data.R * x + data.r)
    eq = _inf_norm(_as_vector(data.J_eq * x + data.b_eq))
    ineq = _inf_norm(min.(_as_vector(data.J_ineq * x + data.b_ineq), 0.0))
    endpoint = max(_inf_norm(min.(a, 0.0)), _inf_norm(min.(b, 0.0)))
    comp = _inf_norm(a .* b)
    return (; eq, ineq, endpoint, comp, kkt_residual)
end

function _is_converged(summary, opts::JuliaMarbleOptions)
    return summary.eq <= opts.convergence_eq_violation &&
           summary.ineq <= opts.convergence_ineq_violation &&
           summary.endpoint <= opts.convergence_ineq_violation &&
           summary.comp <= opts.convergence_comp_violation
end

"""
    solve_julia_marble(data::MarbleData; opts=JuliaMarbleOptions(), z0=nothing)

Experimental pure-Julia solver for QPCCs of the form encoded by `MarbleData`.
For each fixed complementarity relaxation `kappa`, it solves the smooth problem

```
L*x + l = exp(s), R*x + r = exp(t), s + t <= log(kappa)
```

with a log-domain interior-point Newton method for the inequalities, then
reduces `kappa`.
"""
function solve_julia_marble(data::MarbleData; opts::JuliaMarbleOptions=JuliaMarbleOptions(), z0=nothing)
    _check_data_shapes(data)

    start_time = time()
    Q = _as_dense(data.Q)
    q = _as_vector(data.q)
    n = length(q)
    m = length(data.l)
    p = length(data.b_eq) + 2m
    mineq = length(data.b_ineq) + m

    kappa = opts.kappa_initial
    y = _initial_point(data, kappa, z0)
    lambda = zeros(Float64, p)

    rd0 = _residual_data(data, y, lambda, zeros(Float64, mineq), opts.mu_initial, kappa)
    h0 = max.(rd0.h, sqrt(opts.mu_initial))
    v = log.(sqrt(opts.mu_initial) ./ h0)

    total_mu = 0
    total_newton = 0
    last_kkt = Inf
    last_mu = opts.mu_initial
    status = :max_iterations
    outer_used = 0

    for outer in 1:opts.max_outer_iters
        outer_used = outer
        opts.verbosity >= 1 && @printf("kappa %.3e\n", kappa)
        n_mu, n_newton, kkt_residual, mu_used =
            _solve_fixed_kappa!(data, y, lambda, v, kappa, opts)
        total_mu += n_mu
        total_newton += n_newton
        last_kkt = kkt_residual
        last_mu = mu_used

        x = copy(view(y, 1:n))
        summary = _residual_summary(data, x, last_kkt)
        if opts.verbosity >= 1
            @printf("  eq %.3e ineq %.3e endpoint %.3e comp %.3e kkt %.3e\n",
                    summary.eq, summary.ineq, summary.endpoint, summary.comp, summary.kkt_residual)
        end

        if _is_converged(summary, opts)
            status = :converged
            break
        end

        kappa <= opts.kappa_min && break
        kappa = max(kappa * opts.kappa_scaling, opts.kappa_min)

        if m > 0
            _, s, t = _split_z(y, n, m)
            log_kappa = log(max(kappa, eps(Float64)))
            for i in 1:m
                excess = s[i] + t[i] - (log_kappa - 1.0)
                if excess > 0
                    s[i] -= 0.5 * excess
                    t[i] -= 0.5 * excess
                end
            end
        end
    end

    x = copy(view(y, 1:n))
    summary = _residual_summary(data, x, last_kkt)
    if status != :converged && _is_converged(summary, opts)
        status = :converged
    end

    return JuliaMarbleResult(
        x = x,
        status = status,
        objective = Float64(0.5 * dot(x, Q * x) + dot(q, x) + data.c0),
        kappa = kappa,
        mu = last_mu,
        iterations_outer = outer_used,
        iterations_mu = total_mu,
        iterations_newton = total_newton,
        residual_eq = summary.eq,
        residual_ineq = summary.ineq,
        residual_comp = summary.comp,
        residual_endpoint = summary.endpoint,
        kkt_residual = summary.kkt_residual,
        solve_time_s = time() - start_time,
    )
end
