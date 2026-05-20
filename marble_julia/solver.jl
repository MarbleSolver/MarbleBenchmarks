using LinearAlgebra

Base.@kwdef mutable struct LDQPCCState
    x::Vector{Float64}
    ν::Vector{Float64}
    va::Vector{Float64}   # log variable for J_ineq*x + b_ineq >= 0
    vy::Vector{Float64}   # log variable for L*x + l >= 0
    vz::Vector{Float64}   # log variable for R*x + r >= 0
    p::Float64            # log variable for kappa - y'z >= 0
end

Base.@kwdef mutable struct LDQPCCOptions
    method::Symbol = :exact       # :exact or :gn
    μ0::Float64 = 1.0
    μ_min::Float64 = 1e-8
    κ0::Float64 = 1.0
    κ_min::Float64 = 1e-8
    reduction::Float64 = 0.2
    max_outer::Int = 20
    max_inner::Int = 30
    tol::Float64 = 1e-7
    reg::Float64 = 1e-9
    verbose::Bool = true
end

function initial_state(data::MarbleData)
    n  = length(data.q)
    me = length(data.b_eq)
    ma = length(data.b_ineq)
    mc = length(data.l)

    return LDQPCCState(
        x  = zeros(n),
        ν  = zeros(me),
        va = zeros(ma),
        vy = zeros(mc),
        vz = zeros(mc),
        p  = 0.0,
    )
end

function residual_components(data::MarbleData, st::LDQPCCState, μ, κ)
    Q, q = data.Q, data.q
    E, beq = data.J_eq, data.b_eq
    A, bineq = data.J_ineq, data.b_ineq
    L, l = data.L, data.l
    R, r = data.R, data.r

    x = st.x
    sqrtμ = sqrt(μ)

    a = A*x .+ bineq
    y = L*x .+ l
    z = R*x .+ r
    ϕ = κ - dot(y, z)

    â = sqrtμ .* exp.(-st.va)
    ŷ = sqrtμ .* exp.(-st.vy)
    ẑ = sqrtμ .* exp.(-st.vz)
    ϕ̂ = sqrtμ * exp(-st.p)

    λa = sqrtμ .* exp.(st.va)
    λy = sqrtμ .* exp.(st.vy)
    λz = sqrtμ .* exp.(st.vz)
    η  = sqrtμ * exp(st.p)

    gh = L' * z + R' * y

    rstat = Q*x .+ q
    rstat .+= E' * st.ν
    rstat .-= A' * λa
    rstat .-= L' * λy
    rstat .-= R' * λz
    rstat .+= η .* gh

    req = E*x .+ beq
    ra = a .- â
    ry = y .- ŷ
    rz = z .- ẑ
    rϕ = ϕ - ϕ̂

    return (; rstat, req, ra, ry, rz, rϕ,
             a, y, z, ϕ, â, ŷ, ẑ, ϕ̂,
             λa, λy, λz, η, gh)
end

function residual_norm(data::MarbleData, st::LDQPCCState, μ, κ)
    c = residual_components(data, st, μ, κ)
    return norm([c.rstat; c.req; c.ra; c.ry; c.rz; [c.rϕ]])
end

function objective(data::MarbleData, x)
    return 0.5 * dot(x, data.Q*x) + dot(data.q, x) + data.c0
end

function newton_step(data::MarbleData, st::LDQPCCState, μ, κ, opts::LDQPCCOptions)
    Q = Matrix(data.Q)
    E = data.J_eq
    A = data.J_ineq
    L = data.L
    R = data.R

    n  = length(st.x)
    me = length(st.ν)

    c = residual_components(data, st, μ, κ)

    Da = exp.(2 .* st.va)
    Dy = exp.(2 .* st.vy)
    Dz = exp.(2 .* st.vz)

    Hh = L' * R + R' * L

    M = copy(Q)
    M .+= A' * Diagonal(Da) * A
    M .+= L' * Diagonal(Dy) * L
    M .+= R' * Diagonal(Dz) * R

    if opts.method == :exact
        M .+= c.η .* Hh
    elseif opts.method == :gn
        # Gauss-Newton / convexified approximation: drop eta * Hh
    else
        error("opts.method must be :exact or :gn")
    end

    M .+= opts.reg .* I(n)

    # Use Δη, not Δp, so the reduced system is symmetric:
    #
    # [ M   E'   gh ] [Δx ]   [ bx  ]
    # [ E   0    0 ] [Δν ] = [-req ]
    # [ gh' 0  -σ ] [Δη ]   [ rϕ  ]
    #
    # where σ = ϕ̂ / η.
    σ = c.ϕ̂ / c.η

    bx = -c.rstat
    bx .-= A' * (Da .* c.ra)
    bx .-= L' * (Dy .* c.ry)
    bx .-= R' * (Dz .* c.rz)

    N = n + me + 1
    K = zeros(N, N)
    rhs = zeros(N)

    ix = 1:n
    iν = n+1:n+me
    iη = N

    K[ix, ix] .= M
    if me > 0
        K[ix, iν] .= E'
        K[iν, ix] .= E
    end
    K[ix, iη] .= c.gh
    K[iη, ix] .= c.gh
    K[iη, iη] = -σ

    rhs[ix] .= bx
    if me > 0
        rhs[iν] .= -c.req
    end
    rhs[iη] = c.rϕ

    sol = K \ rhs

    Δx = sol[ix]
    Δν = me > 0 ? sol[iν] : Float64[]
    Δη = sol[iη]

    Δp = Δη / c.η

    Δva = -(c.ra .+ A*Δx) ./ c.â
    Δvy = -(c.ry .+ L*Δx) ./ c.ŷ
    Δvz = -(c.rz .+ R*Δx) ./ c.ẑ

    return (; Δx, Δν, Δva, Δvy, Δvz, Δp)
end

function add_step(st::LDQPCCState, d, α)
    return LDQPCCState(
        x  = st.x  .+ α .* d.Δx,
        ν  = st.ν  .+ α .* d.Δν,
        va = st.va .+ α .* d.Δva,
        vy = st.vy .+ α .* d.Δvy,
        vz = st.vz .+ α .* d.Δvz,
        p  = st.p  + α * d.Δp,
    )
end

function copy_state(st::LDQPCCState)
    return LDQPCCState(
        x  = copy(st.x),
        ν  = copy(st.ν),
        va = copy(st.va),
        vy = copy(st.vy),
        vz = copy(st.vz),
        p  = st.p,
    )
end

function solve_ld_qpcc(data::MarbleData; opts::LDQPCCOptions = LDQPCCOptions(),
                       state::LDQPCCState = initial_state(data))

    st = state
    μ = opts.μ0
    κ = opts.κ0

    history = NamedTuple[]

    for outer in 1:opts.max_outer
        for inner in 1:opts.max_inner
            rn = residual_norm(data, st, μ, κ)
            push!(history, (; state=copy_state(st), μ, κ))

            if opts.verbose
                @info "LD-QPCC" outer inner opts.method μ κ rn obj=objective(data, st.x)
            end

            if rn < opts.tol && μ <= opts.μ_min && κ <= opts.κ_min
                return (; x=st.x, state=st, μ, κ, residual=rn,
                        objective=objective(data, st.x), history)
            end

            d = newton_step(data, st, μ, κ, opts)

            # Basic merit backtracking on ||F||.
            f0 = 0.5 * rn^2
            α = 1.0
            accepted = false

            for _ in 1:25
                trial = add_step(st, d, α)
                rn_trial = residual_norm(data, trial, μ, κ)
                f_trial = 0.5 * rn_trial^2

                if isfinite(f_trial) && f_trial <= (1 - 1e-4 * α) * f0
                    st = trial
                    accepted = true
                    break
                end

                α *= 0.5
            end

            if !accepted
                # Take a tiny damped step anyway.
                st = add_step(st, d, α)
            end
        end

        μ = max(opts.reduction * μ, opts.μ_min)
        κ = max(opts.reduction * κ, opts.κ_min)
    end

    rn = residual_norm(data, st, μ, κ)
    return (; x=st.x, state=st, μ, κ, residual=rn,
            objective=objective(data, st.x), history)
end
