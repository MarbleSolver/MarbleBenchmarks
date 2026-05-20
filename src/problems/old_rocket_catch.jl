module RocketCatch2D
export create_problem

using CILogDomain
using LinearAlgebra
using CairoMakie
using ControlSystems
import ForwardDiff as FD
using FileIO
using ColorSchemes


function rocket_cdynamics(x, u; m, g, l, J)
    R(θ) = [cos(θ) -sin(θ);
            sin(θ)  cos(θ)]

    px, py, vx, vy, θ, ω, α = x
    T, α̇ = u

    a_f = 1/m * R(θ + α) * [0; T]

    return [
        vx;
        vy;
        a_f[1];
        a_f[2] - g; #+ 1/m * λ;
        ω;
        1/J * (-l * T * sin(α));
        α̇
    ]
end

function create_problem(; N = 40, T = 5.0)
    problem_data, traj_indices, nz = begin
        nx = 7
        nu = 2

        # Problem parameters
        tower_height = 3.0
        catch_height = 1.5
        catch_arm_length = 1

        glideslope_angle = 30.0

        s = tan(deg2rad(glideslope_angle))
        
        glideslope_A = [1 0]
        glideslope_b = [0.0]

        traj_indices, nz = indexed_components((
            x = (nx, N),
            u = (nu, N-1),
            # STC trigger : y <= tower_height
            gp = N, # g+ for tower height constraint
            gn = N, # g- for tower height constraint
            hp = (3, N), # h+ for A*p >= b, θ ≥ 0
            hn = (3, N), # h- for A*p >= b, θ ≥ 0
        ))

        # linearization point
        m, g = 1.0, 9.81
        l = 0.5 # half-length of rocket
        J = 1/12 * m * (2*l)^2

        x̄ = zeros(7)
        ū = [m*g, 0]

        @assert rocket_cdynamics(x̄, ū; m=m, g=g, l=l, J=J) ≈ zeros(nx)

        A = FD.jacobian(x -> rocket_cdynamics(x, ū; m=m, g=g, l=l, J=J), x̄)
        B = FD.jacobian(u -> rocket_cdynamics(x̄, u; m=m, g=g, l=l, J=J), ū)
        c = rocket_cdynamics(x̄, ū; m=m, g=g, l=l, J=J)

        Δt = T / (N - 1)
        sysc = ss(A, [B c], I(nx), zeros(nx, nu+1))
        sysd = c2d(sysc, Δt, :zoh)

        problem_data = (
            N = N,
            T = T,
            Δt = Δt,
            nx = nx,
            nu = nu,
            x̄ = x̄,
            ū = ū,
            l = l,
            g = g,
            m = m,
            J = J,
            catch_height = catch_height,
            tower_height = tower_height,
            catch_arm_length = catch_arm_length,
            θ_max = 0.2,
            ω_max = 0.6,
            α_max = 0.5,
            T_max = 12.0,
            α̇_max = 0.8,
            glideslope_angle = glideslope_angle,
            glideslope_A = glideslope_A,
            glideslope_b = glideslope_b,
            Ad = sysd.A,
            Bd = sysd.B[:, 1:nu],
            cd = sysd.B[:, end],
            xinit = [
                10.0, 20.0,  # initial position
                -3.0, -8.0,  # initial velocity
                0.15, 0.2,  # initial angle, angular velocity
                0.0        # initial gimbal angle
            ],
        )

        (problem_data, traj_indices, nz)
    end

    function cost_fn(k::Int)        
        i_u = k < N ? traj_indices.u[k] : nothing
        i_α = traj_indices.x[k][7]
        
        i_gp, i_gn = traj_indices.gp[k], traj_indices.gn[k]
        i_hp, i_hn = traj_indices.hp[k], traj_indices.hn[k]

        w = 1

        cost_func(params, z::AbstractVector) = begin
            cost = 0
            
            cost += 1/2 * params.Δt * z[i_α]^2
            cost += params.Δt * w * (z[i_gp] + z[i_gn] + sum(z[i_hp]) + sum(z[i_hn]))

            if k < N
                cost += 1/2 * params.Δt * (params.ū[1] + z[i_u[1]])^2
            end

            return cost
        end
        
        cost_grad(params, z::AbstractVector, grad::AbstractVector) = begin
            grad[i_α] += params.Δt * z[i_α]

            grad[i_gp] += params.Δt * w
            grad[i_gn] += params.Δt * w
            grad[i_hp] .+= params.Δt * w
            grad[i_hn] .+= params.Δt * w

            if k < N
                grad[i_u[1]] += params.Δt * (params.ū[1] + z[i_u[1]])
            end
        end

        cost_hess(params, z::AbstractVector, hess::AbstractMatrix) = begin
            hess[i_α, i_α] += params.Δt

            if k < N
                hess[i_u[1], i_u[1]] += params.Δt
            end
        end

        return (cost_func=cost_func, cost_grad=cost_grad, cost_hess=cost_hess)
    end

    function initial_conditions()
        i_x1 = traj_indices.x[1]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= z[i_x1] .- params.xinit;
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_x1] .= I(problem_data.nx)
        sparsity(conjac::AbstractMatrix) = conjac[:, i_x1] .= 1

        return (length=problem_data.nx,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, problem_data.nx))
    end

    function terminal_conditions()
        i_xN = traj_indices.x[N]
        
        H = [
            1 0 0 0 0 0 0;
           -1 0 0 0 0 0 0;
            0 1 0 0 0 0 0;
            0 0 1 0 0 0 0;
            0 0 0 1 0 0 0;
            0 0 0 0 1 0 0;
            # 0 0 0 0 0 1 0
        ]

        f = [
            0.0; # x ≥ 0
            -catch_arm_length;
            problem_data.catch_height; 
            0.0; # vx = 0
            0.0; # vy = 0
            0.0; # θ = 0
            # 0.0  # ω = 0
        ]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= H * z[i_xN] - f
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_xN] .= H
        sparsity(conjac::AbstractMatrix) = conjac[:, i_xN] .= 1

        return (length=length(f),
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=[
                    Ineq::ConstraintType,
                    Ineq::ConstraintType,
                    Eq::ConstraintType,
                    Eq::ConstraintType,
                    Eq::ConstraintType,
                    Eq::ConstraintType,
                    # Eq::ConstraintType,
                ])
    end

    function dynamics(k::Int)
        i_x = traj_indices.x[k]
        i_xnext = traj_indices.x[k+1]
        i_u = traj_indices.u[k]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= z[i_xnext] - (params.Ad * z[i_x] + params.Bd * z[i_u] + params.cd)
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = begin
            conjac[:, i_xnext] .= I(problem_data.nx)
            conjac[:, i_x] .= -params.Ad
            conjac[:, i_u] .= -params.Bd
        end
        sparsity(conjac::AbstractMatrix) = begin
            conjac[:, i_xnext] .= 1
            conjac[:, i_x] .= 1
            conjac[:, i_u] .= 1
        end

        return (length=problem_data.nx,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Eq::ConstraintType, problem_data.nx))
    end

    function stc(k::Int)
        i_p = traj_indices.x[k][1:2]
        _, i_y = i_p
        i_θ = traj_indices.x[k][5]
        i_α = traj_indices.x[k][7]

        i_gp, i_gn = traj_indices.gp[k], traj_indices.gn[k]
        i_hp, i_hn = traj_indices.hp[k], traj_indices.hn[k]

        g(params, y) = 1.3*params.tower_height - y
        dg(params) = -1
        dg_sparsity() = 1

        J = [0 -1;
             1 0]
        e1 = [1, 0]
        r_top_b    = [0; problem_data.l]
        r_bottom_b = [0; -problem_data.l]

        h(params, θ, α, p) = begin
            R_wb = I(2) + θ * J

            [
                θ + α,
                e1' * (p + R_wb * r_top_b),
                e1' * (p + R_wb * r_bottom_b),
            ]
        end

        dh(params) = begin
            return [
                1 1 0 0;
                e1' * J * r_top_b 0 e1'
                e1' * J * r_bottom_b 0 e1'
            ]
        end

        dh_sparsity() = [
            1 1 0 0;
            1 0 1 1;
            1 0 1 1;
        ]
        
        ng, nh = 1, length(i_hp)

        sizes = [ng, nh, 2ng, 2nh, 2]
        types = [
            fill(Eq::ConstraintType, ng)...,        # g+ - g- = g
            fill(Eq::ConstraintType, nh)...,        # h+ - h- = h
            fill(Ineq::ConstraintType, 2ng)...,     # g- ≥ 0
            fill(Ineq::ConstraintType, 2nh)...,     # h+ ≥ 0
            fill(Comp::ConstraintType, 2)...,       # complementarity
        ]

        ends   = cumsum(sizes)
        starts = ends .- sizes .+ 1
        ranges = collect.(UnitRange.(starts, ends))

        residual(params, z::AbstractVector, con::AbstractVector) = begin
            con[ranges[1]] .= g(params, z[i_y]) - (z[i_gp] - z[i_gn])
            con[ranges[2]] .= h(params, z[i_θ], z[i_α], z[i_p]) - (z[i_hp] - z[i_hn])
            con[ranges[3]] .= [z[i_gp]; z[i_gn]]
            con[ranges[4]] .= [z[i_hp]; z[i_hn]]
            con[ranges[5]] .= [z[i_gp], sum(z[i_hn])]
        end

        function jacobian(params, z::AbstractVector, conjac::AbstractMatrix)
            # g constraints
            conjac[ranges[1], [i_gp; i_gn]] .= [-I(ng) I(ng)]
            conjac[ranges[1], [i_y]] .= dg(params)

            # h constraints
            conjac[ranges[2], [i_hp; i_hn]] .= [-I(nh) I(nh)]
            conjac[ranges[2], [i_θ; i_α; i_p]] .= dh(params)

            # g ≥ 0
            conjac[ranges[3], [i_gp; i_gn]] .= I(2*ng)

            # h ≥ 0
            conjac[ranges[4], [i_hp; i_hn]] .= I(2*nh)

            # complementarity
            conjac[ranges[5][1], i_gp] = 1
            conjac[ranges[5][2], i_hn] .= ones(nh)
        end

        function sparsity(conjac::AbstractMatrix)
            # g constraints
            conjac[ranges[1], [i_gp; i_gn]] .= [I(ng) I(ng)]
            conjac[ranges[1], [i_y]] .= dg_sparsity()

            # h constraints
            conjac[ranges[2], [i_hp; i_hn]] .= [I(nh) I(nh)]
            conjac[ranges[2], [i_θ; i_α; i_p]] .= dh_sparsity()

            # g ≥ 0
            conjac[ranges[3], [i_gp; i_gn]] .= I(2*ng)

            # h ≥ 0
            conjac[ranges[4], [i_hp; i_hn]] .= I(2*nh)

            # complementarity
            conjac[ranges[5][1], i_gp] = 1
            conjac[ranges[5][2], i_hn] .= ones(nh)
        end

        return (length=sum(sizes),
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=types)
    end

    function control_bounds(k::Int)
        i_u = traj_indices.u[k]

        w_lo = [-problem_data.ū[1], -problem_data.α̇_max]
        w_hi = [problem_data.T_max - problem_data.ū[1], problem_data.α̇_max]
        H = [I(problem_data.nu);
            -I(problem_data.nu)]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= H * z[i_u] - [w_lo; -w_hi]
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_u] .= H
        sparsity(conjac::AbstractMatrix) = conjac[:, i_u] .= 1

        return (length=2*problem_data.nu,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Ineq::ConstraintType, 2*problem_data.nu))
    end

    function state_bounds(k::Int)
        i_x = traj_indices.x[k]

        w_lo = -[problem_data.θ_max, problem_data.ω_max, problem_data.α_max]
        w_hi =  [problem_data.θ_max, problem_data.ω_max, problem_data.α_max]

        H1 = [zeros(3, problem_data.nx - 3) I(3)]
        H = [H1;
            -H1]

        residual(params, z::AbstractVector, con::AbstractVector) = con .= H * z[i_x] - [w_lo; -w_hi]
        jacobian(params, z::AbstractVector, conjac::AbstractMatrix) = conjac[:, i_x] .= H
        sparsity(conjac::AbstractMatrix) = conjac[:, i_x] .= 1

        return (length=2*3,
                residual=residual,
                jacobian=jacobian,
                sparsity=sparsity,
                type=fill(Ineq::ConstraintType, 2*3))
    end

    
    costs = Vector{NamedTuple}([
        [cost_fn(k) for k in 1:problem_data.N]...,
    ])

    constraints = Vector{NamedTuple}([
        initial_conditions(),
        terminal_conditions(),
        [dynamics(k) for k in 1:problem_data.N-1]...,
        [stc(k) for k in 1:problem_data.N]...,
        [control_bounds(k) for k in 1:problem_data.N-1]...,
        [state_bounds(k) for k in 1:problem_data.N]...,
    ])

    return Problem(problem_data,
                nz,
                costs,
                constraints; validate=true, traj_indices=traj_indices)
end

end # module RocketCatch