import ForwardDiff as FD
using LinearAlgebra, SparseArrays
using Plots
using Marble, MarbleBenchmarks


@inline function bκ(t, κ)
    s = sqrt(t*t + 4κ)
    return ifelse(t >= zero(t), 0.5 * (t + s), 2κ / (s - t))
end

@inline function dbκ(t, κ)
    # Equivalent to 0.5 * (1 + t / sqrt(t^2 + 4κ)),
    # but numerically consistent with the stable bκ.
    return bκ(t, κ) / sqrt(t*t + 4κ)
end

function block_ranges(sizes)
    offsets = cumsum([0; sizes])
    return [offsets[i] + 1 : offsets[i + 1] for i in eachindex(sizes)]
end

function build_indices(nvar, neq, nineq, ncc)
    full = block_ranges([nvar, neq, nineq, ncc, ncc])
    reduced = block_ranges([nvar, neq, nineq, ncc])

    inds = (;
        x = full[1],
        λ = full[2],
        u = full[3],
        v = full[4],
        z = full[5],
    )

    rinds = (;
        x = reduced[1],
        λ = reduced[2],
        u = reduced[3],
        z = reduced[4],
    )

    return (; full = inds, reduced = rinds)
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
        inds,
    )
end

function split_full(sol, data)
    I = data.inds.full
    return (;
        x = sol[I.x],
        λ = sol[I.λ],
        u = sol[I.u],
        v = sol[I.v],
        z = sol[I.z],
    )
end

function kkt_diagonals(u, v, z, κ)
    b_u⁺ = bκ.( u, κ)
    b_u⁻ = bκ.(-u, κ)

    b_v⁺ = bκ.( v, κ)
    b_v⁻ = bκ.(-v, κ)

    B_u⁺ = Diagonal(dbκ.( u, κ))
    B_u⁻ = Diagonal(dbκ.(-u, κ))

    B_v⁺ = Diagonal(dbκ.( v, κ))
    B_v⁻ = Diagonal(dbκ.(-v, κ))

    P_v⁺ = Diagonal(b_v⁺)
    P_v⁻ = Diagonal(b_v⁻)

    Z = Diagonal(z)

    return (;
        b_u⁺, b_u⁻,
        b_v⁺, b_v⁻,
        B_u⁺, B_u⁻,
        B_v⁺, B_v⁻,
        P_v⁺, P_v⁻,
        Z,
    )
end

function kkt_residual(sol, data, κ)
    (; Q, q, A, b, G, h, L, l, R, r) = data
    (; x, λ, u, v, z) = split_full(sol, data)

    d = kkt_diagonals(u, v, z, κ)

    r_x =
        Q*x + q +
        A'*λ -
        G'*d.b_u⁻ +
        L'*(z .* d.b_v⁻) +
        R'*(z .* d.b_v⁺)

    r_A = A*x + b
    r_G = G*x + h - d.b_u⁺
    r_L = L*x + l - d.b_v⁺
    r_R = R*x + r - d.b_v⁻

    return (; r_x, r_A, r_G, r_L, r_R)
end

function residual_vector(sol, data, κ)
    r = kkt_residual(sol, data, κ)
    return [r.r_x; r.r_A; r.r_G; r.r_L; r.r_R]
end
function kkt_residual_penalty(sol, data, κ, ρeq)
    (; Q, q, A, b, G, h, L, l, R, r) = data
    (; x, λ, u, v, z) = split_full(sol, data)

    d = kkt_diagonals(u, v, z, κ)
    r_A = A * x + b

    # Equality-only AL shifted multiplier:
    #   λhat = λ + ρ(Ax+b)
    λhat = λ + ρeq * r_A

    r_x =
        Q*x + q +
        A' * λhat -
        G'*d.b_u⁻ +
        L'*(z .* d.b_v⁻) +
        R'*(z .* d.b_v⁺)

    r_G = G*x + h - d.b_u⁺
    r_L = L*x + l - d.b_v⁺
    r_R = R*x + r - d.b_v⁻

    return (; r_x, r_A, r_G, r_L, r_R)
end

function residual_vector_penalty(sol, data, κ, ρeq)
    r = kkt_residual_penalty(sol, data, κ, ρeq)
    return [r.r_x; r.r_G; r.r_L; r.r_R]
end


function kkt_system_full_fd(sol, data, κ)
    r = residual_vector(sol, data, κ)
    J = FD.jacobian(y -> residual_vector(y, data, κ), sol)
    return (; r, J)
end

function kkt_system_full(sol, data, κ)
    (; Q, A, G, L, R, nvar, neq, nineq, ncc) = data
    (; u, v, z) = split_full(sol, data)

    If = data.inds.full

    d = kkt_diagonals(u, v, z, κ)
    r = residual_vector(sol, data, κ)

    J_xv = -L' * d.Z * d.B_v⁻ + R' * d.Z * d.B_v⁺
    J_xz = L' * d.P_v⁻ + R' * d.P_v⁺

    J = zeros(nvar + neq + nineq + 2ncc, nvar + neq + nineq + 2ncc)

    J[If.x, If.x] = Q
    J[If.x, If.λ] = A'
    J[If.x, If.u] = G' * d.B_u⁻
    J[If.x, If.v] = J_xv
    J[If.x, If.z] = J_xz

    J[If.λ, If.x] = A

    J[If.u, If.x] = G
    J[If.u, If.u] = -d.B_u⁺

    J[If.v, If.x] = L
    J[If.v, If.v] = -d.B_v⁺

    J[If.z, If.x] = R
    J[If.z, If.v] = d.B_v⁻

    return (; r, J)
end


# Scaled reduced KKT system
function kkt_system_reduced_scaled(sol, data, κ)
    (; Q, A, G, L, R, nvar, neq, nineq, ncc) = data
    (; u, v, z) = split_full(sol, data)

    Ir = data.inds.reduced

    d = kkt_diagonals(u, v, z, κ)
    r = kkt_residual(sol, data, κ)

    # Transformed residuals
    r̃_x =
        r.r_x -
        G' * r.r_G +
        R' * d.Z * r.r_L +
        L' * d.Z * r.r_R

    r̃_C =
        d.P_v⁻ * r.r_L +
        d.P_v⁺ * r.r_R

    r̃_D =
        r.r_L - r.r_R

    # T_v = P_v⁺ + P_v⁻, stored as a vector of diagonal entries.
    T_v = d.b_v⁺ + d.b_v⁻

    # Scaled fourth residual block: T_v^{-1} r̃_C.
    residual = [
        r̃_x;
        r.r_A;
        r.r_G;
        r̃_C ./ T_v
    ]

    # Scaled symmetric blocks:
    #   Hκ = Q - G'G + L'ZR + R'ZL
    #   J̄κ = B_v⁻ L + B_v⁺ R
    Hκ =
        Q -
        G' * G +
        L' * d.Z * R +
        R' * d.Z * L

    J̄κ =
        d.B_v⁻ * L +
        d.B_v⁺ * R

    J = zeros(nvar + neq + nineq + ncc, nvar + neq + nineq + ncc)

    J[Ir.x, Ir.x] = Hκ
    J[Ir.x, Ir.λ] = A'
    J[Ir.x, Ir.u] = G'
    J[Ir.x, Ir.z] = J̄κ'

    J[Ir.u, Ir.u] = -d.B_u⁺
    
    return (; r = residual, J = Symmetric(triu(J)), r̃_D, T_v)
end

function kkt_system_reduced_scaled_penalty(sol, data, κ, ρeq)
    (; Q, A, G, L, R, nvar, nineq, ncc) = data
    (; u, v, z) = split_full(sol, data)

    Ip = (; zip((:x, :u, :z), block_ranges([nvar, nineq, ncc]))...)

    d = kkt_diagonals(u, v, z, κ)
    r = kkt_residual_penalty(sol, data, κ, ρeq)

    r̃_x =
        r.r_x -
        G' * r.r_G +
        R' * d.Z * r.r_L +
        L' * d.Z * r.r_R

    r̃_C =
        d.P_v⁻ * r.r_L +
        d.P_v⁺ * r.r_R

    r̃_D =
        r.r_L - r.r_R

    T_v = d.b_v⁺ + d.b_v⁻

    residual = [
        r̃_x;
        r.r_G;
        r̃_C ./ T_v
    ]

    Hκ =
        Q +
        ρeq * A' * A -
        G' * G +
        L' * d.Z * R +
        R' * d.Z * L

    J̄κ =
        d.B_v⁻ * L +
        d.B_v⁺ * R

    J = zeros(nvar + nineq + ncc, nvar + nineq + ncc)

    J[Ip.x, Ip.x] = Hκ
    J[Ip.x, Ip.u] = G'
    J[Ip.x, Ip.z] = J̄κ'

    J[Ip.u, Ip.u] = -d.B_u⁺

    return (; r = residual, J = Symmetric(triu(J)), r̃_D, T_v, pinds = Ip)
end


function solve_kkt_system_full(sol, data, κ; check_condition = false)
    sys = kkt_system_full(sol, data, κ)

    if check_condition
        @info "[full direct] cond(J) = $(cond(sys.J))"
    end

    return sys.J \ (-sys.r)
end



# Solve scaled reduced KKT system
function solve_kkt_system_reduced_scaled(sol, data, κ; check_condition = false)
    sys = kkt_system_reduced_scaled(sol, data, κ)

    if check_condition
        @info "[reduced scaled] cond(J) = $(cond(sys.J))"
    end

    Δred = sys.J \ (-sys.r)

    Ir = data.inds.reduced
    If = data.inds.full

    Δx = Δred[Ir.x]
    Δλ = Δred[Ir.λ]
    Δu = Δred[Ir.u]

    # The scaled system solves for Δζ = T_v Δz.
    # Recover Δz = T_v^{-1} Δζ elementwise.
    Δζ = Δred[Ir.z]
    Δz = Δζ ./ sys.T_v

    # Difference row:
    #   (L - R)Δx - Δv = -r̃_D
    # therefore
    #   Δv = (L - R)Δx + r̃_D
    Δv = (data.L - data.R) * Δx + sys.r̃_D

    Δfull = similar(Δred, length(If.x) + length(If.λ) + length(If.u) + length(If.v) + length(If.z))

    Δfull[If.x] = Δx
    Δfull[If.λ] = Δλ
    Δfull[If.u] = Δu
    Δfull[If.v] = Δv
    Δfull[If.z] = Δz

    return Δfull
end

function solve_kkt_system_reduced_scaled_penalty(sol, data, κ, ρeq; check_condition = false)
    sys = kkt_system_reduced_scaled_penalty(sol, data, κ, ρeq)

    if check_condition
        @info "[penalty reduced scaled] cond(J) = $(cond(sys.J))"
    end

    Δred = sys.J \ (-sys.r)

    Ip = sys.pinds
    If = data.inds.full

    Δx = Δred[Ip.x]
    Δu = Δred[Ip.u]

    Δζ = Δred[Ip.z]
    Δz = Δζ ./ sys.T_v

    Δv = (data.L - data.R) * Δx + sys.r̃_D

    Δfull = zeros(length(If.x) + length(If.λ) + length(If.u) + length(If.v) + length(If.z))
    Δfull[If.x] = Δx
    Δfull[If.u] = Δu
    Δfull[If.v] = Δv
    Δfull[If.z] = Δz

    return Δfull
end

function backtracking_line_search(
    sol,
    Δ,
    data,
    κ;
    α0 = 1.0,
    ρ = 0.5,
    c = 1e-4,
    α_min = 1e-12,
)
    r0 = residual_vector(sol, data, κ)
    φ0 = 0.5 * dot(r0, r0)

    α = α0

    while α >= α_min
        sol_trial = sol + α * Δ
        r_trial = residual_vector(sol_trial, data, κ)
        φ_trial = 0.5 * dot(r_trial, r_trial)

        if isfinite(φ_trial) && φ_trial <= (1 - c * α) * φ0
            return α
        end

        α *= ρ
    end

    @error "Line search failed" α_min=α_min
    return α_min
end

function backtracking_line_search_penalty(
    sol,
    Δ,
    data,
    κ,
    ρeq;
    α0 = 1.0,
    ρ = 0.5,
    c = 1e-4,
    α_min = 1e-12,
)
    r0 = residual_vector_penalty(sol, data, κ, ρeq)
    φ0 = 0.5 * dot(r0, r0)

    α = α0

    while α >= α_min
        sol_trial = sol + α * Δ
        r_trial = residual_vector_penalty(sol_trial, data, κ, ρeq)
        φ_trial = 0.5 * dot(r_trial, r_trial)

        if isfinite(φ_trial) && φ_trial <= (1 - c * α) * φ0
            return α
        end

        α *= ρ
    end

    @error "Penalty line search failed" α_min=α_min
    return α_min
end


function residual_report(sol, data, κ)
    r = kkt_residual(sol, data, κ)

    println("||Qx + q + A'λ - G'b(-u) + L'(z ⊙ b(-v)) + R'(z ⊙ b(v))||∞ = ", norm(r.r_x, Inf))
    println("||Ax + b||∞ = ", norm(r.r_A, Inf))
    println("||Gx + h - b(u)||∞ = ", norm(r.r_G, Inf))
    println("||Lx + l - b(v)||∞ = ", norm(r.r_L, Inf))
    println("||Rx + r - b(-v)||∞ = ", norm(r.r_R, Inf))
    println("||kkt_residual||∞   = ", norm(residual_vector(sol, data, κ), Inf))
end

function residual_report_penalty(sol, data, κ, ρeq)
    r = kkt_residual_penalty(sol, data, κ, ρeq)

    println("||penalty stationarity||∞ = ", norm(r.r_x, Inf))
    println("||Ax + b||∞              = ", norm(r.r_A, Inf))
    println("||Gx + h - b(u)||∞       = ", norm(r.r_G, Inf))
    println("||Lx + l - b(v)||∞       = ", norm(r.r_L, Inf))
    println("||Rx + r - b(-v)||∞      = ", norm(r.r_R, Inf))
    println("||penalty residual||∞    = ", norm(residual_vector_penalty(sol, data, κ, ρeq), Inf))
end


function solve_fixed_κ(sol, data, κ; max_iters=50, tol=1e-10)
    for iter in 1:max_iters
        r = residual_vector(sol, data, κ)
        nr = norm(r, Inf)

        if nr < tol
            return sol
        end

        Δ = solve_kkt_system_reduced_scaled(sol, data, κ; check_condition=true)

        # Diagnostic: compare the reduced scaled step against the full direct Newton system.
        full = kkt_system_full(sol, data, κ)
        newton_defect = norm(full.J * Δ + full.r, Inf)
        α = backtracking_line_search(sol, Δ, data, κ)

        @info "Iter stats $iter" newton_defect=newton_defect α=α κ=κ residual_inf=nr residual_2=norm(r)

        sol = sol + α * Δ
    end

    return sol
end
function solve_fixed_κ_penalty(
    sol,
    data,
    κ,
    ρeq;
    max_iters = 50,
    tol = 1e-10,
    eq_tol = 1e-8 + 5 * sqrt(κ),
    update_multiplier = true,
)
    If = data.inds.full

    for iter in 1:max_iters
        r = kkt_residual_penalty(sol, data, κ, ρeq)
        rvec = residual_vector_penalty(sol, data, κ, ρeq)

        nr = norm(rvec, Inf)
        eq_norm = norm(r.r_A, Inf)

        # Inner convergence is governed ONLY by the inner residual
        #   (r_x, r_G, r_L, r_R).
        # Ax+b is NOT an inner Newton equation; it is reduced by the outer
        # multiplier update below, so it must not gate the inner solve. (The
        # √κ equality floor eq_tol is a target for the OUTER loop, not a cap on
        # the inner residual, which is fully solvable to tol at fixed κ.)
        @info "Penalty iter stats $iter" κ=κ ρeq=ρeq residual_inf=nr residual_2=norm(rvec) equality_inf=eq_norm λ_inf=norm(sol[If.λ], Inf)
        
        if nr < tol
            break
        end

        Δ = solve_kkt_system_reduced_scaled_penalty(
            sol,
            data,
            κ,
            ρeq;
            check_condition = true,
        )

        α = backtracking_line_search_penalty(sol, Δ, data, κ, ρeq)

        # λ is held fixed during the inner solve (Δ's λ block is identically 0).
        sol = sol + α * Δ
    end

    # Outer equality-only AL multiplier update: λ⁺ = λ + ρ(Ax⁺ + b).
    #
    # Gate the update on the finite-κ homotopy floor ε_A(κ) = eq_tol. The
    # smoothed complementarity manifold (b_κ(v)·b_κ(-v) = κ > 0) is generally
    # incompatible with exact Ax+b = 0, so ‖Ax+b‖ bottoms out at an O(√κ) floor.
    # Once the equality sits at that floor, bumping λ no longer reduces the
    # residual — it only inflates λ and, through stationarity, the
    # complementarity multiplier z. Past the floor the right move is to
    # decrease κ, not to keep updating λ.
    if update_multiplier
        eq_res = data.A * split_full(sol, data).x .+ data.b
        if norm(eq_res, Inf) > eq_tol
            sol[If.λ] .+= ρeq .* eq_res
        end
    end

    return sol
end

function simple_test1()
    Q = 2.0 * Matrix(1.0I, 4, 4)
    q = zeros(4)
    c0 = 0.0
    A, b = [1.0 0.0 0.0 0.0], [-1.0]
    G, h = [0.0 1.0 0.0 0.0], [-1.0]
    L, l = [0.0 0.0 1.0 0.0], [1.0]
    R, r = [0.0 0.0 0.0 1.0], [-1.0]

    data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)
    sol = zeros(data.nvar + data.neq + data.nineq + 2 * data.ncc)

    return data, sol
end

function simple_test2()
    # min x'x
    # st  x[1] + x[2] = 0
    #     0 ≤ x[1] ⊥ x[2] ≥ 0
    #
    # At the exact zero/biactive point v = 0, the hard-equality KKT columns for
    # λ and z are collinear: stationarity only sees λ + sqrt(κ) z. Use the
    # penalty KKT path below when you want the relaxed complementarity equations
    # prioritized and Ax+b treated as a soft residual.

    Q = 2.0 * Matrix(1.0I, 2, 2)
    q = zeros(2)
    c0 = 0.0
    A, b = [1.0 1.0], [0.0]
    G, h = zeros(0, 2), zeros(0)
    L, l = [1.0 0.0], [0.0]
    R, r = [0.0 1.0], [0.0]

    data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)
    sol = zeros(data.nvar + data.neq + data.nineq + 2 * data.ncc)

    return data, sol
end

function simple_test3()
    # min x'x
    # st  x[1] + x[2] = 1
    #     0 ≤ x[1] ⊥ x[2] ≥ 0
    
    Q = 2.0 * Matrix(1.0I, 2, 2)
    q = zeros(2)
    c0 = 0.0
    A, b = [1.0 1.0], [-1.0]
    G, h = zeros(0, 2), zeros(0)
    L, l = [1.0 0.0], [0.0]
    R, r = [0.0 1.0], [0.0]

    data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)
    sol = randn(data.nvar + data.neq + data.nineq + 2 * data.ncc)

    return data, sol
end

function macmpec(name)
    problem = get_problem(name)
    Q, q, c0, A, b, G, h, L, l, R, r = jump_to_marble(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)
    data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)
    sol = zeros(data.nvar + data.neq + data.nineq + 2 * data.ncc)
    return data, sol
end

##
data, sol = simple_test1()
sol = randn(length(sol))

Δfull = solve_kkt_system_full(sol, data, 1e-2; check_condition=true)
Δreduced = solve_kkt_system_reduced_scaled(sol, data, 1e-2; check_condition=true)

@assert Δfull ≈ Δreduced

##
data, sol = simple_test2()
ρeq = 1e2
κ = 1e-1

sol = solve_fixed_κ_penalty(sol, data, κ, ρeq; max_iters=25, tol=1e-10)

##
data, sol = macmpec(Symbol("bard1"))
ρeq = 1e6
κ = 1e-6

sol = solve_fixed_κ_penalty(sol, data, κ, ρeq; max_iters=100, tol=1e-8)
x = sol[data.inds.full.x]

nothing
# ## MacMPEC / benchmark test
# using NLPModelsJuMP
# using Marble, MarbleBenchmarks

# # data, sol = simple_test_residualnz()
# data, sol = macmpec(Symbol("ex9.2.2"))

# κ = 1e-1
# sol = solve_fixed_κ(sol, data, κ; max_iters=100, tol=1e-8)

# x = sol[data.inds.full.x]
# cc = (data.L * x + data.l) .* (data.R * x + data.r)
# obj = 0.5 * x' * data.Q * x + data.q' * x + data.c0

# @show norm(residual_vector(sol, data, κ), Inf)
# @show norm(data.A * x + data.b, Inf)
# @show norm(cc, Inf)
# @show obj

# ## Manual one-step test with the reduced scaled system
# sys = kkt_system_reduced_scaled(sol, data, κ)
# Δ_reduced_scaled = solve_kkt_system_reduced_scaled(sol, data, κ; check_condition = true)

# full = kkt_system_full(sol, data, κ)
# @show norm(full.J * Δ_reduced_scaled + full.r, Inf)

# α = backtracking_line_search(sol, Δ_reduced_scaled, data, κ)
# sol = sol + α * Δ_reduced_scaled

# @show norm(residual_vector(sol, data, κ), Inf)
# x = sol[data.inds.full.x]
# cc = (data.L * x + data.l) .* (data.R * x + data.r)
# @show norm(cc, Inf)

# nothing
# # ## MacMPEC tets
# # using MarbleBenchmarks
# # using Marble

# # name = Symbol("bard1m")
# # problem = get_problem(name)
# # Q, q, c0, A, b, G, h, L, l, R, r = jump_to_marble(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)

# # data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)
# # sol = zeros(data.nvar + data.neq + data.nineq + 2 * data.ncc)
# # κ = 1e-2

# # # sol[data.inds.full.x] .= [1.0, 0.0, 9.0, 12.5, 0.0, 0.0]

# ##

# sys = kkt_system_reduced_scaled(sol, data, κ)

# sol = solve_fixed_κ(sol, data, κ; max_iters=100, tol=1e-8)

# x = sol[data.inds.full.x]
# v = sol[data.inds.full.v]
# cc = (data.L * x + data.l) .* (data.R * x + data.r)
# obj = 0.5 * x' * data.Q * x + data.q' * x + data.c0

# ##
# κ = κ / 10

# ##

# # κ = 1.0

# # Δ_full = solve_kkt_system_full(sol, data, κ; check_condition = true)
# # Δ_reduced = solve_kkt_system_reduced(sol, data, κ; check_condition = true)
# Δ_reduced_scaled = solve_kkt_system_reduced_scaled(sol, data, κ; check_condition = true)

# # @assert Δ_full ≈ Δ_reduced ≈ Δ_reduced_scaled

# α = backtracking_line_search(sol, Δ_reduced_scaled, data, κ)
# sol = sol + α * Δ_reduced_scaled

# @show norm(residual_vector(sol, data, κ))
# x = sol[data.inds.full.x]
# cc = (data.L * x + data.l) .* (data.R * x + data.r)

# nothing

# # x = split_full(sol, data).x
# # obj = 0.5 * x' * data.Q * x + data.q' * x + data.c0

# # @show norm(residual_vector(sol, data, κ))

# # residual_report(sol, data, κ)
# # reduced_residual_report(sol, data, κ)
