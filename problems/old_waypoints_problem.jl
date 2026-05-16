using JuMP, NLPModelsJuMP
using LinearAlgebra
using BlockDiagonals
import ForwardDiff as FD
using ControlSystems

# Quaternion helpers
function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end

function L(q)
    s = q[1]
    v = q[2:4]
    L = [s    -v';
         v  s*I+hat(v)]
    return L
end

T = Diagonal([1; -ones(3)])
H = [zeros(1,3); I]

function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end

function G(q)
    G = L(q)*H
end

function E(q)
    E = BlockDiagonal([1.0*I(3), G(q), 1.0*I(3)])
end

function rptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[1; ϕ]
end

function qtorp(q)
    q[2:4]/q[1]
end

function quad_dynamics(x, u; m, g)
    r = x[1:3]
    q = x[4:7]
    v = x[8:10]

    T = u[1]
    ω = u[2:4]
    
    Q = qtoQ(q)
    
    ṙ = v
    q̇ = 0.5*L(q)*H*ω
    
    v̇ = [0; 0; -g] + (1/m) * Q * ([0, 0, 1]*T)
    
    return [ṙ; q̇; v̇]
end

function linearized_quad_dynamics(x̄, ū, Δt; m, g)
    nx, nu = 10, 4
    r0, q0, v0 = x̄[1:3], x̄[4:7], x̄[8:10]

    Afull = FD.jacobian(x -> quad_dynamics(x, ū; m=m, g=g), x̄) # df / dx
    Bfull = FD.jacobian(u -> quad_dynamics(x̄, u; m=m, g=g), ū) # df / du
    cfull = quad_dynamics(x̄, ū; m=m, g=g)

    sysc = ss(Afull, [Bfull cfull], I(nx), zeros(nx, nu+1))
    sysd = c2d(sysc, Δt, :zoh)

    Adfull = sysd.A
    Bdfull = sysd.B

    Ãd = E(q0)'*Adfull*E(q0)
    B̃d = E(q0)'*Bdfull;

    return Ãd, B̃d[:, 1:nu], B̃d[:, end]
end

function create_problem(; N=30, T=7.0)
    nx = 3 + 4 + 3  # pos, quat, vel
    nx̃ = 3 + 3 + 3  # pos, rp, vel
    nu = 1 + 3      # thrust + angular rates
    
    ngates = 4

    traj_indices, nz = indexed_components((
        x = (nx̃, N),   # pos, vel
        u = (nu, N-1), # accel
        e     = (ngates, 3, N),
        s_pos = (ngates, 3, N),
        s_neg = (ngates, 3, N),
        λ = (ngates, N), # waypoint completion state
        μ = (ngates, N), # waypoint completion control
    ))

    problem_data = begin
        Δt = T/(N-1)

        m, g = 1.0, 9.81
        q0 = [1; 0; 0; 0]
        x̄ = [zeros(3); q0; zeros(3)]
        ū = [m*g; zeros(3)]

        Ãd, B̃d, c̃d = linearized_quad_dynamics(x̄, ū, Δt; m=m, g=g)

        (
            Δt = Δt,
            N = N,
            T = T,
            # dynamics
            x̄ = x̄,
            ū = ū,
            Ãd = Ãd,
            B̃d = B̃d,
            c̃d = c̃d,
            nx = nx,
            nx̃ = nx̃,
            nu = nu,
            xstart = [zeros(3); qtorp(q0); zeros(3)],
            # state / control bounds
            Tmin = 0.0*m*g,
            Tmax = 4.0*m*g,
            ωmin = [-2, -2, -1.5],
            ωmax = [2, 2, 1.5],
            ϕmin = deg2rad.([-20, -20, -20]),
            ϕmax = deg2rad.([20, 20, 20]),
            # gate info
            ngates = ngates,
            gate_positions = [[-2, 2, 0.25],
                              [1, 6, 1],
                              [2, 3, 2],
                              [-4, 6, 0.5]],
            gate_orientations = [
                so3(0, 0, deg2rad(45)),
                so3(0, 0, deg2rad(-45)),
                so3(0, 0, 0),
                so3(0, 0, deg2rad(20)),
            ],
            gate_dims = [1, 0, 1],  # gates are rectangles in xy plane with given width (x), height (y), depth (z)
        )
    end

    function cost_fn(k::Int)        
        i_uk = k < N ? traj_indices.u[k] : nothing
        i_s_pos = traj_indices.s_pos[k]
        i_s_neg = traj_indices.s_neg[k]

        cost_func(params, z::AbstractVector) = begin
            cost = 0

            if k < N
                u = params.ū + z[i_uk]
                cost += 1/2 * params.Δt * u'*u
            end

            for j in 1:params.ngates
                cost += params.Δt * sum(z[i_s_pos[j]] + z[i_s_neg[j]])
            end

            return cost
        end
        
        cost_grad(params, z::AbstractVector, grad::AbstractVector) = begin
            if k < N
                u = params.ū + z[i_uk]
                grad[i_uk] .+= params.Δt * u
            end

            for j in 1:params.ngates
                grad[i_s_pos[j]] .+= params.Δt
                grad[i_s_neg[j]] .+= params.Δt
            end
        end

        cost_hess(params, z::AbstractVector, hess::AbstractMatrix) = begin
            if k < N
                hess[i_uk, i_uk] .+= params.Δt * I(length(i_uk))
            end
        end

        return (cost_func=cost_func, cost_grad=cost_grad, cost_hess=cost_hess)
    end

    function dynamics_constraint(k::Int)
        i_x = traj_indices.x[k]
        i_xnext = traj_indices.x[k+1]
        i_u = traj_indices.u[k]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= z[i_xnext] - (params.Ãd * z[i_x] + params.B̃d * z[i_u] + params.c̃d)
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = begin
            conjac[:, i_xnext] .= I(params.nx̃)
            conjac[:, i_x] .= -params.Ãd
            conjac[:, i_u] .= -params.B̃d
        end
        sparsity(conjac::AbstractMatrix) = begin
            conjac[:, i_xnext] .= 1
            conjac[:, i_x] .= 1
            conjac[:, i_u] .= 1
        end

        return (length=problem_data.nx̃,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, problem_data.nx̃))
    end

    function state_bounds(k::Int)
        i_x = traj_indices.x[k]
        H = [zeros(3, 3) I(3) zeros(3, 3);
             zeros(3, 3) -I(3) zeros(3, 3)]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= H * z[i_x] - [params.ϕmin; -params.ϕmax]
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_x] .= H
        sparsity(conjac::AbstractMatrix) = conjac[:, i_x] .= abs.(H)

        return (length=6,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Ineq::ConstraintType, 6))
    end

    function control_bounds(k::Int)
        i_u = traj_indices.u[k]

        T0 = problem_data.ū[1]
        H = [I(4);
             -I(4)]
        
        umin = [problem_data.Tmin - T0; problem_data.ωmin]
        umax = [problem_data.Tmax - T0; problem_data.ωmax]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= H*z[i_u] - [umin; -umax]
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_u] .= H
        sparsity(conjac::AbstractMatrix) = conjac[:, i_u] .= abs.(H)

        return (length=8,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Ineq::ConstraintType, 8))
    end

    function state_boundary_conditions()
        i_x1 = traj_indices.x[1]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= z[i_x1] - params.xstart;
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_x1] .= I(problem_data.nx̃)
        sparsity(conjac::AbstractMatrix) = conjac[:, i_x1] .= 1

        return (length=problem_data.nx̃,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, problem_data.nx̃))
    end

    function state_terminal_conditions()
        i_xN = traj_indices.x[end]

        # H = [zeros(6, 3) I(6)]
        H = I(problem_data.nx̃)
        # xf = [0, 9, 0]
        vf = zeros(3) # not moving
        qf = qtorp([1, 0, 0, 0])  # level orientation

        xf = [[0, 10, 0]; qf; vf]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= H * z[i_xN] - xf;
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_xN] .= H
        sparsity(conjac::AbstractMatrix) = conjac[:, i_xN] .= abs.(H)

        return (length=length(xf),
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, length(xf)))
    end

    function waypoint_boundary_conditions()
        i_λN = traj_indices.λ[end]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= z[i_λN] .- 1.0;
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_λN] .= I(ngates)
        sparsity(conjac::AbstractMatrix) = conjac[:, i_λN] .= 1

        return (length=ngates,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, ngates))
    end

    function waypoint_dynamics(k::Int)
        i_λ = traj_indices.λ[k]
        i_μ = traj_indices.μ[k]
        i_λprev = k > 1 ? traj_indices.λ[k-1] : nothing

        residual(params, z::AbstractVector, con::AbstractVector) = begin
            con .= k == 1 ? z[i_λ] - z[i_μ] : z[i_λ] - (z[i_λprev] + z[i_μ])
        end
        
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = begin
            if k == 1
                conjac[:, i_λ] .= I(ngates)
                conjac[:, i_μ] .= -I(ngates)
            else
                conjac[:, i_λ]     .= I(ngates)
                conjac[:, i_λprev] .= -I(ngates)
                conjac[:, i_μ]     .= -I(ngates)
            end
        end

        sparsity(conjac::AbstractMatrix) = begin
            if k == 1
                conjac[:, i_λ] .= I(ngates)
                conjac[:, i_μ] .= I(ngates)
            else
                conjac[:, i_λ]     .= I(ngates)
                conjac[:, i_λprev] .= I(ngates)
                conjac[:, i_μ]     .= I(ngates)
            end
        end

        return (length=ngates,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, ngates))
    end

    function waypoint_completion_constraints(k::Int, j::Int)
        i_x = traj_indices.x[k]
        i_r = i_x[1:3]

        i_s_pos = traj_indices.s_pos[k][j]
        i_s_neg = traj_indices.s_neg[k][j]
        i_e = traj_indices.e[k][j]
        i_μ = traj_indices.μ[k][j]
        
        nq = 3
        lengths = [
            nq,  # e
            nq,  # e bounds
            nq,  # e bounds
            nq,  # s_neg
            nq,  # s_pos
            1,   # μ
            1,   # sum slacks
        ]

        ids = cumsum([0; lengths])
        ids = [(ids[i]+1):ids[i+1] for i in 1:length(lengths)]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= [
                # Definition of e vector
                z[i_e] - (params.gate_orientations[j]' * (z[i_r] - params.gate_positions[j]));
                # Bounds on e for gate passage when inside oriented bounding box
                z[i_e] + 1/2 * params.gate_dims + z[i_s_neg];
                -z[i_e] + 1/2 * params.gate_dims + z[i_s_pos];
                # Non-negative slacks
                z[i_s_neg];
                z[i_s_pos];
                # Pass through allowed when inside gate OBB
                z[i_μ];
                sum(z[i_s_pos] + z[i_s_neg])
            ]

        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = begin
            conjac[ids[1], i_r] .= -params.gate_orientations[j]'
            conjac[ids[1], i_e] .= I(nq)

            conjac[ids[2], i_e] .= I(nq)
            conjac[ids[2], i_s_neg] .= I(nq)

            conjac[ids[3], i_e] .= -I(nq)
            conjac[ids[3], i_s_pos] .= I(nq)

            conjac[ids[4], i_s_neg] .= I(nq)
            conjac[ids[5], i_s_pos] .= I(nq)

            conjac[ids[6], i_μ] .= 1.0
            conjac[ids[7], [i_s_pos; i_s_neg]] .= 1.0
        end

        sparsity(conjac::AbstractMatrix) = begin
            conjac[ids[1], i_x[1:nq]] .= 1
            conjac[ids[1], i_e] .= I(nq)

            conjac[ids[2], i_e] .= I(nq)
            conjac[ids[2], i_s_neg] .= I(nq)

            conjac[ids[3], i_e] .= -I(nq)
            conjac[ids[3], i_s_pos] .= I(nq)

            conjac[ids[4], i_s_neg] .= I(nq)
            conjac[ids[5], i_s_pos] .= I(nq)

            conjac[ids[6], i_μ] .= 1
            conjac[ids[7], [i_s_pos; i_s_neg]] .= 1
        end

        return (length=sum(lengths),
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=[
                    fill(Eq::ConstraintType, nq)...,    # e
                    fill(Ineq::ConstraintType, nq)...,  # e bounds
                    fill(Ineq::ConstraintType, nq)...,  # e bounds
                    fill(Ineq::ConstraintType, nq)...,  # s_neg
                    fill(Ineq::ConstraintType, nq)...,  # s_pos
                    Comp::ConstraintType,               # μ
                    Comp::ConstraintType,               # sum slacks
                ])
    end

    function waypoint_inorder_traversal(k::Int)
        i_λ = traj_indices.λ[k]
        A = diagm(0 => ones(ngates-1), 1 => -ones(ngates-1))[1:end-1, :]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= A * z[i_λ]
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_λ] .= A
        sparsity(conjac::AbstractMatrix) = conjac[:, i_λ] .= 1

        return (length=ngates-1,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Ineq::ConstraintType, ngates-1))
    end

    costs = Vector{NamedTuple}([
        [cost_fn(k) for k in 1:N]...,
    ])

    constraints = Vector{NamedTuple}([
        state_boundary_conditions(),
        state_terminal_conditions(),
        [dynamics_constraint(k) for k in 1:N-1]...,
        [state_bounds(k) for k in 1:N]...,
        [control_bounds(k) for k in 1:N-1]...,
        waypoint_boundary_conditions(),
        [waypoint_dynamics(k) for k in 1:N]...,
        [waypoint_completion_constraints(k, j) for k in 1:N, j in 1:ngates]...,
        # Optional: enforce waypoint traversal order
        [waypoint_inorder_traversal(k) for k in 1:N]...,
    ])

    return Problem(problem_data,
                   nz,
                   costs,
                   constraints; validate=true, traj_indices=traj_indices)
end
