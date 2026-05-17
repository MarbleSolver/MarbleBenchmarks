using LinearAlgebra
using SparseArrays
using Printf
using QDLDL

Base.@kwdef mutable struct JuliaMarbleOptions
    convergence_kkt_norm::Float64 = 1e-4
    convergence_eq_violation::Float64 = 1e-4
    convergence_ineq_violation::Float64 = 1e-4
    convergence_comp_violation::Float64 = 1e-5
    outer_step_kkt_norm::Float64 = 1e-6
    penalty_initial::Float64 = 10.0
    penalty_max::Float64 = 1e6
    penalty_scaling::Float64 = 10.0
    relaxation_initial::Float64 = 1e-1
    relaxation_min::Float64 = 1e-7
    relaxation_scaling::Float64 = 0.5
    max_iters::Int = 1000
    max_iters_linesearch::Int = 10
    gamma_objective::Float64 = 1e-5
    gamma_constraint::Float64 = 1e-5
    ruiz_iterations::Int = 10
    verbosity::Int = 0
    print_every::Int = 1
    max_regularization::Float64 = 1e8
    theta_improvement_factor::Float64 = 0.75
    inertia_correction::Bool = true
end

Base.@kwdef struct JuliaMarbleResult
    converged::Bool
    iterations::Int
    iterations_outer::Int
    iterations_inner::Int
    factorizations::Int
    z::Vector{Float64}
    s_ineq::Vector{Float64}
    s_comp::Vector{Float64}
    m_eq::Vector{Float64}
    m_ineq::Vector{Float64}
    m_comp::Vector{Float64}
    kkt::Float64
    eq_violation::Float64
    ineq_violation::Float64
    comp_violation::Float64
    obj::Float64
    history::Vector{NamedTuple}
end

struct _JuliaMarbleProblem
    Q::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
    c0::Float64
    J_eq::SparseMatrixCSC{Float64,Int}
    b_eq::Vector{Float64}
    J_ineq::SparseMatrixCSC{Float64,Int}
    b_ineq::Vector{Float64}
    J_comp::SparseMatrixCSC{Float64,Int}
    c_comp::Vector{Float64}
    n::Int
    n_eq::Int
    n_ineq::Int
    n_comp::Int
end

mutable struct _JuliaMarbleState
    x::Vector{Float64}
    m_eq_est::Vector{Float64}
    m_ineq_est::Vector{Float64}
    m_comp_est::Vector{Float64}
    residual_eq::Vector{Float64}
    residual_ineq::Vector{Float64}
    residual_comp::Vector{Float64}
    kkt_residual::Vector{Float64}
    scaling::Vector{Float64}
    penalty::Float64
    relaxation::Float64
end

struct _FilterEntry
    feas::Float64
    merit::Float64
end

mutable struct _Filter
    entries::Vector{_FilterEntry}
end

_Filter() = _Filter(_FilterEntry[])

function _as_problem(data)
    Q = sparse(Float64.(data.Q))
    q = Float64.(data.q)
    n = length(q)
    J_eq = sparse(Float64.(data.J_eq))
    b_eq = Float64.(data.b_eq)
    J_ineq = sparse(Float64.(data.J_ineq))
    b_ineq = Float64.(data.b_ineq)
    L = sparse(Float64.(data.L))
    l = Float64.(data.l)
    R = sparse(Float64.(data.R))
    r = Float64.(data.r)

    n_comp = length(l)
    J_comp = spzeros(Float64, 2 * n_comp, n)
    c_comp = zeros(Float64, 2 * n_comp)
    for i in 1:n_comp
        J_comp[2 * i - 1, :] = L[i, :]
        J_comp[2 * i, :] = R[i, :]
        c_comp[2 * i - 1] = l[i]
        c_comp[2 * i] = r[i]
    end

    return _JuliaMarbleProblem(
        Q, q, Float64(data.c0),
        J_eq, b_eq,
        J_ineq, b_ineq,
        sparse(J_comp), c_comp,
        n, length(b_eq), length(b_ineq), n_comp,
    )
end

function _ranges(p::_JuliaMarbleProblem)
    n, ni, nc, ne = p.n, p.n_ineq, p.n_comp, p.n_eq
    z = 1:n
    s_ineq = (last(z) + 1):(last(z) + ni)
    s_comp = (last(s_ineq) + 1):(last(s_ineq) + nc)
    m_eq = (last(s_comp) + 1):(last(s_comp) + ne)
    m_ineq = (last(m_eq) + 1):(last(m_eq) + ni)
    m_comp = (last(m_ineq) + 1):(last(m_ineq) + 2 * nc)
    return (; z, s_ineq, s_comp, m_eq, m_ineq, m_comp)
end

_r(s, sqrtκ) = sqrtκ * 0.5 * (s / sqrtκ + sqrt((s / sqrtκ)^2 + 4.0))
_dr(s, sqrtκ) = 0.5 * (s / sqrtκ / sqrt((s / sqrtκ)^2 + 4.0) + 1.0)
_ddr(s, sqrtκ) = (2.0 / ((s / sqrtκ)^2 + 4.0)^(1.5)) / sqrtκ

_r(v::AbstractVector, sqrtκ) = _r.(v, sqrtκ)
_dr(v::AbstractVector, sqrtκ) = _dr.(v, sqrtκ)
_ddr(v::AbstractVector, sqrtκ) = _ddr.(v, sqrtκ)

function _objective(p::_JuliaMarbleProblem, z)
    return 0.5 * dot(z, p.Q * z) + dot(p.q, z) + p.c0
end

function _compute_residuals!(st::_JuliaMarbleState, p::_JuliaMarbleProblem)
    idx = _ranges(p)
    z = st.x[idx.z]
    st.residual_eq = p.J_eq * z + p.b_eq
    st.residual_ineq = p.J_ineq * z + p.b_ineq
    st.residual_comp = p.J_comp * z + p.c_comp
    return nothing
end

function _update_kkt_residual!(st::_JuliaMarbleState, p::_JuliaMarbleProblem)
    idx = _ranges(p)
    sqrtκ = sqrt(st.relaxation)
    invρ = inv(st.penalty)
    z = st.x[idx.z]
    s_ineq = st.x[idx.s_ineq]
    s_comp = st.x[idx.s_comp]
    m_eq = st.x[idx.m_eq]
    m_ineq = st.x[idx.m_ineq]
    m_comp = st.x[idx.m_comp]

    _compute_residuals!(st, p)

    st.kkt_residual .= 0.0
    st.kkt_residual[idx.z] =
        p.Q * z + p.q +
        p.J_eq' * m_eq +
        p.J_ineq' * m_ineq +
        p.J_comp' * m_comp

    if p.n_ineq > 0
        p_neg_ineq = _r(-s_ineq, sqrtκ)
        d_p_ineq = _dr(s_ineq, sqrtκ)
        st.kkt_residual[idx.s_ineq] = d_p_ineq .* (-m_ineq .- p_neg_ineq)
        st.kkt_residual[idx.m_ineq] =
            st.residual_ineq .- (s_ineq .+ p_neg_ineq) .- invρ .* (m_ineq .- st.m_ineq_est)
    end

    if p.n_comp > 0
        d_p_comp = _dr(s_comp, sqrtκ)
        p_comp = _r(s_comp, sqrtκ)
        for i in 1:p.n_comp
            st.kkt_residual[idx.s_comp[i]] =
                -d_p_comp[i] * m_comp[2 * i - 1] + (1.0 - d_p_comp[i]) * m_comp[2 * i]
            st.kkt_residual[idx.m_comp[2 * i - 1]] =
                st.residual_comp[2 * i - 1] - p_comp[i] - invρ * (m_comp[2 * i - 1] - st.m_comp_est[2 * i - 1])
            st.kkt_residual[idx.m_comp[2 * i]] =
                st.residual_comp[2 * i] - (p_comp[i] - s_comp[i]) - invρ * (m_comp[2 * i] - st.m_comp_est[2 * i])
        end
    end

    if p.n_eq > 0
        st.kkt_residual[idx.m_eq] = st.residual_eq .- invρ .* (m_eq .- st.m_eq_est)
    end
    return nothing
end

function _build_kkt(st::_JuliaMarbleState, p::_JuliaMarbleProblem; regularization=0.0)
    idx = _ranges(p)
    N = length(st.x)
    K = spzeros(Float64, N, N)
    sqrtκ = sqrt(st.relaxation)
    invρ = inv(st.penalty)

    K[idx.z, idx.z] = p.Q
    if regularization != 0.0
        for i in idx.z
            K[i, i] += regularization
        end
    end

    if p.n_eq > 0
        K[idx.z, idx.m_eq] = p.J_eq'
        K[idx.m_eq, idx.z] = p.J_eq
        for i in idx.m_eq
            K[i, i] = -invρ
        end
    end

    if p.n_ineq > 0
        d = _dr(st.x[idx.s_ineq], sqrtκ)
        sstat = .-d .* d .+ d
        K[idx.z, idx.m_ineq] = p.J_ineq'
        K[idx.m_ineq, idx.z] = p.J_ineq
        for (j, row) in enumerate(idx.s_ineq)
            K[row, row] = sstat[j] + regularization
            K[row, idx.m_ineq[j]] = -d[j]
            K[idx.m_ineq[j], row] = -d[j]
            K[idx.m_ineq[j], idx.m_ineq[j]] = -invρ
        end
    end

    if p.n_comp > 0
        s = st.x[idx.s_comp]
        m = st.x[idx.m_comp]
        d = _dr(s, sqrtκ)
        dd = _ddr(s, sqrtκ)
        K[idx.z, idx.m_comp] = p.J_comp'
        K[idx.m_comp, idx.z] = p.J_comp
        for i in 1:p.n_comp
            srow = idx.s_comp[i]
            m1 = idx.m_comp[2 * i - 1]
            m2 = idx.m_comp[2 * i]
            K[srow, srow] = -dd[i] * (m[2 * i - 1] + m[2 * i]) + regularization
            K[srow, m1] = -d[i]
            K[m1, srow] = -d[i]
            K[srow, m2] = 1.0 - d[i]
            K[m2, srow] = 1.0 - d[i]
            K[m1, m1] = -invρ
            K[m2, m2] = -invρ
        end
    end

    return sparse(K)
end

function _linfty_by_col!(out, A::SparseMatrixCSC)
    fill!(out, 0.0)
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in 1:size(A, 2)
        for ptr in nzrange(A, j)
            out[j] = max(out[j], abs(vals[ptr]))
        end
    end
    return out
end

function _linfty_by_row!(out, A::SparseMatrixCSC)
    fill!(out, 0.0)
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in 1:size(A, 2)
        for ptr in nzrange(A, j)
            out[rows[ptr]] = max(out[rows[ptr]], abs(vals[ptr]))
        end
    end
    return out
end

function _ruiz_scaling(p::_JuliaMarbleProblem, niter::Int)
    idx = _ranges(p)
    N = last(idx.m_comp)
    scaling = ones(Float64, N)
    n, ne, ni, nc = p.n, p.n_eq, p.n_ineq, 2 * p.n_comp

    Hbar = copy(p.Q)
    Jeq = copy(p.J_eq)
    Jineq = copy(p.J_ineq)
    Jcomp = copy(p.J_comp)

    d = view(scaling, idx.z)
    eeq = view(scaling, idx.m_eq)
    eineq = view(scaling, idx.m_ineq)
    ecomp = view(scaling, idx.m_comp)

    dtmp = zeros(Float64, n)
    eqtmp = zeros(Float64, ne)
    ineqtmp = zeros(Float64, ni)
    comptmp = zeros(Float64, nc)

    for _ in 1:niter
        _linfty_by_col!(dtmp, Hbar)
        if ne > 0
            dtmp .= max.(dtmp, vec(maximum(abs.(Jeq), dims=1)))
            _linfty_by_row!(eqtmp, Jeq)
        end
        if ni > 0
            dtmp .= max.(dtmp, vec(maximum(abs.(Jineq), dims=1)))
            _linfty_by_row!(ineqtmp, Jineq)
        end
        if nc > 0
            dtmp .= max.(dtmp, vec(maximum(abs.(Jcomp), dims=1)))
            _linfty_by_row!(comptmp, Jcomp)
        end

        dstep = 1.0 ./ sqrt.(clamp.(dtmp, 1e-4, 1e4))
        eqstep = ne == 0 ? Float64[] : 1.0 ./ sqrt.(clamp.(eqtmp, 1e-4, 1e4))
        ineqstep = ni == 0 ? Float64[] : 1.0 ./ sqrt.(clamp.(ineqtmp, 1e-4, 1e4))
        compstep = nc == 0 ? Float64[] : 1.0 ./ sqrt.(clamp.(comptmp, 1e-4, 1e4))

        Dstep = spdiagm(0 => dstep)
        Hbar = Dstep * Hbar * Dstep
        if ne > 0
            Jeq = spdiagm(0 => eqstep) * Jeq * Dstep
            eeq .*= eqstep
        end
        if ni > 0
            Jineq = spdiagm(0 => ineqstep) * Jineq * Dstep
            eineq .*= ineqstep
        end
        if nc > 0
            Jcomp = spdiagm(0 => compstep) * Jcomp * Dstep
            ecomp .*= compstep
        end
        d .*= dstep
    end

    view(scaling, idx.s_ineq) .= 1.0
    view(scaling, idx.s_comp) .= 1.0
    return scaling
end

function _filter_acceptable(filter::_Filter, cand::_FilterEntry, opts::JuliaMarbleOptions)
    isempty(filter.entries) && return true
    return all(filter.entries) do entry
        feas_ok = cand.feas <= (1.0 - opts.gamma_constraint) * entry.feas
        merit_ok = cand.merit <= entry.merit - opts.gamma_objective * entry.feas
        feas_ok || merit_ok
    end
end

function _filter_update!(filter::_Filter, cand::_FilterEntry, opts::JuliaMarbleOptions)
    keep = _FilterEntry[]
    for entry in filter.entries
        feas_ok = entry.feas <= (1.0 - opts.gamma_constraint) * cand.feas
        merit_ok = entry.merit <= cand.merit - opts.gamma_objective * cand.feas
        (feas_ok || merit_ok) && push!(keep, entry)
    end
    push!(keep, cand)
    filter.entries = keep
    return nothing
end

function _filter_entry(st::_JuliaMarbleState, p::_JuliaMarbleProblem)
    idx = _ranges(p)
    sqrtκ = sqrt(st.relaxation)
    invρ = inv(st.penalty)
    z = st.x[idx.z]
    p_ineq = _r(st.x[idx.s_ineq], sqrtκ)
    p_comp = _r(st.x[idx.s_comp], sqrtκ)

    m_eq_feas = st.residual_eq .- invρ .* (st.x[idx.m_eq] .- st.m_eq_est)
    m_ineq_feas = st.residual_ineq .- p_ineq .- invρ .* (st.x[idx.m_ineq] .- st.m_ineq_est)
    m_comp_feas = copy(st.residual_comp .- invρ .* (st.x[idx.m_comp] .- st.m_comp_est))
    for i in 1:p.n_comp
        m_comp_feas[2 * i - 1] -= p_comp[i]
        m_comp_feas[2 * i] -= p_comp[i] - st.x[idx.s_comp[i]]
    end

    n_duals = p.n_eq + p.n_ineq + 2 * p.n_comp
    feas = n_duals == 0 ? 0.0 : (norm(m_eq_feas, 1) + norm(m_ineq_feas, 1) + norm(m_comp_feas, 1)) / n_duals
    barrier = p.n_ineq == 0 ? 0.0 : st.relaxation * sum(log.(max.(p_ineq, eps())))
    merit = _objective(p, z) - barrier + 0.5 * invρ * (
        norm(st.x[idx.m_eq])^2 + norm(st.x[idx.m_ineq])^2 + norm(st.x[idx.m_comp])^2
    )
    return _FilterEntry(feas, merit)
end

function _metrics(st::_JuliaMarbleState, p::_JuliaMarbleProblem)
    idx = _ranges(p)
    _compute_residuals!(st, p)
    kkt = norm(st.kkt_residual, Inf)
    eq_vio = p.n_eq == 0 ? 0.0 : norm(st.residual_eq, Inf)
    ineq_vio = p.n_ineq == 0 ? 0.0 : norm(min.(st.residual_ineq, 0.0), Inf)
    comp_vio = p.n_comp == 0 ? 0.0 : norm(st.residual_comp[1:2:end] .* st.residual_comp[2:2:end], Inf)
    obj = _objective(p, st.x[idx.z])
    return (; kkt, eq_vio, ineq_vio, comp_vio, obj)
end

function _theta(st::_JuliaMarbleState, p::_JuliaMarbleProblem)
    eq_vio = p.n_eq == 0 ? 0.0 : norm(st.residual_eq, Inf)
    ineq_vio = p.n_ineq == 0 ? 0.0 : norm(min.(st.residual_ineq, 0.0), Inf)
    comp_nonneg = 0.0
    comp_product = 0.0
    if p.n_comp > 0
        left = st.residual_comp[1:2:end]
        right = st.residual_comp[2:2:end]
        comp_nonneg = max(norm(min.(left, 0.0), Inf), norm(min.(right, 0.0), Inf))
        comp_product = norm(left .* right, Inf)
    end
    return max(eq_vio, ineq_vio, comp_nonneg, comp_product)
end

function _line_search!(st::_JuliaMarbleState, p::_JuliaMarbleProblem, step, filter::_Filter, opts)
    x0 = copy(st.x)
    α = 1.0
    for _ in 1:opts.max_iters_linesearch
        st.x .= x0 .+ α .* step
        _compute_residuals!(st, p)
        cand = _filter_entry(st, p)
        if _filter_acceptable(filter, cand, opts)
            _filter_update!(filter, cand, opts)
            return true
        end
        α *= 0.5
    end
    st.x .= x0
    _compute_residuals!(st, p)
    return false
end

function _solve_step(st::_JuliaMarbleState, p::_JuliaMarbleProblem, opts::JuliaMarbleOptions)
    reg = 0.0
    attempts = 0
    n_primals = p.n + p.n_ineq + p.n_comp
    n_duals = p.n_eq + p.n_ineq + 2 * p.n_comp
    while reg <= opts.max_regularization
        K = _build_kkt(st, p; regularization=reg)
        D = spdiagm(0 => st.scaling)
        Kscaled = sparse(D * K * D)
        try
            attempts += 1
            F = QDLDL.qdldl(Kscaled)
            if opts.inertia_correction
                try
                    Dvals = F.workspace.D
                    n_pos = count(>(1e-10), Dvals)
                    n_neg = count(<(-1e-10), Dvals)
                    if (n_pos, n_neg) != (n_primals, n_duals)
                        reg = reg == 0.0 ? 1e-8 : 10.0 * reg
                        continue
                    end
                catch
                    # QDLDL.jl has changed its exposed factorization fields
                    # across versions. This reference implementation keeps going
                    # if inertia is not inspectable.
                end
            end
            y = QDLDL.solve(F, -(st.scaling .* st.kkt_residual))
            return st.scaling .* y, reg, attempts
        catch err
            reg = reg == 0.0 ? 1e-8 : 10.0 * reg
        end
    end
    return nothing, reg, attempts
end

"""
    solve_julia_marble(data; opts=JuliaMarbleOptions(), initial_z=nothing)

Pure-Julia reference implementation of the Marble relaxed AL solver. `data` is
expected to be a `Marble.MarbleData`-compatible object with fields `Q`, `q`,
`c0`, `J_eq`, `b_eq`, `J_ineq`, `b_ineq`, `L`, `l`, `R`, and `r`.
"""
function solve_julia_marble(data; opts::JuliaMarbleOptions=JuliaMarbleOptions(), initial_z=nothing)
    p = _as_problem(data)
    idx = _ranges(p)
    N = last(idx.m_comp)
    x = zeros(Float64, N)
    if initial_z !== nothing
        x[idx.z] .= Float64.(initial_z)
    end

    st = _JuliaMarbleState(
        x,
        zeros(p.n_eq), zeros(p.n_ineq), zeros(2 * p.n_comp),
        zeros(p.n_eq), zeros(p.n_ineq), zeros(2 * p.n_comp),
        zeros(N),
        _ruiz_scaling(p, opts.ruiz_iterations),
        opts.penalty_initial,
        opts.relaxation_initial,
    )

    filter = _Filter()
    history = NamedTuple[]
    n_outer = 0
    n_inner = 0
    n_fact = 0
    θ_prev = Inf
    outer_kkt_adjustment = 1.0
    last_outer_iter = -1
    converged = false

    for iter in 0:(opts.max_iters - 1)
        _update_kkt_residual!(st, p)
        met = _metrics(st, p)
        push!(history, (; iter, step_type=:check, penalty=st.penalty, relaxation=st.relaxation, met...))

        if opts.verbosity > 0 && iter % opts.print_every == 0
            @printf("%5d  ρ=%8.1e κ=%8.1e  kkt=%9.2e eq=%9.2e ineq=%9.2e comp=%9.2e obj=%9.2e\n",
                iter, st.penalty, st.relaxation, met.kkt, met.eq_vio, met.ineq_vio, met.comp_vio, met.obj)
        end

        if met.kkt < opts.convergence_kkt_norm &&
           met.eq_vio < opts.convergence_eq_violation &&
           met.ineq_vio < opts.convergence_ineq_violation &&
           met.comp_vio < opts.convergence_comp_violation
            converged = true
            break
        end

        if norm(st.kkt_residual, Inf) < outer_kkt_adjustment * opts.outer_step_kkt_norm
            n_outer += 1
            if iter > 0 && last_outer_iter == iter - 1 && st.relaxation <= opts.relaxation_min
                outer_kkt_adjustment /= 10.0
            end

            θ = _theta(st, p)
            if θ < opts.theta_improvement_factor * θ_prev
                st.m_eq_est .= st.x[idx.m_eq]
                st.m_ineq_est .= st.x[idx.m_ineq]
                st.m_comp_est .= st.x[idx.m_comp]
                st.relaxation = max(st.relaxation * opts.relaxation_scaling, opts.relaxation_min)
                θ_prev = θ
            else
                st.penalty = min(st.penalty * opts.penalty_scaling, opts.penalty_max)
            end
            empty!(filter.entries)
            last_outer_iter = iter
        else
            n_inner += 1
            step, reg, attempts = _solve_step(st, p, opts)
            n_fact += attempts
            step === nothing && error("Julia Marble reference solver could not factor a regularized KKT system")
            ok = _line_search!(st, p, step, filter, opts)
            ok || error("Julia Marble reference solver line search failed")
        end
    end

    _update_kkt_residual!(st, p)
    met = _metrics(st, p)
    return JuliaMarbleResult(
        converged=converged,
        iterations=n_outer + n_inner,
        iterations_outer=n_outer,
        iterations_inner=n_inner,
        factorizations=n_fact,
        z=copy(st.x[idx.z]),
        s_ineq=copy(st.x[idx.s_ineq]),
        s_comp=copy(st.x[idx.s_comp]),
        m_eq=copy(st.x[idx.m_eq]),
        m_ineq=copy(st.x[idx.m_ineq]),
        m_comp=copy(st.x[idx.m_comp]),
        kkt=met.kkt,
        eq_violation=met.eq_vio,
        ineq_violation=met.ineq_vio,
        comp_violation=met.comp_vio,
        obj=met.obj,
        history=history,
    )
end
