using JuMP, NLPModelsJuMP
using NLPModels
using MPCCModels
using AmplNLReader


function multi_robot_coordination2(; N=30, T=5.0)
    Δt = T / (N - 1)
    m = 1.0
    d_min = 2.5

    x0A = [-5, 0, 0, 0]
    x0B = [5, 0, 0, 0]
    xfA = [5, 10, 0, 0]
    xfB = [-5, 10, 0, 0]

    model = JuMP.Model()

    @variable(model, xA[1:N, 1:4])
    @variable(model, uA[1:N-1, 1:2])
    @variable(model, xB[1:N, 1:4])
    @variable(model, uB[1:N-1, 1:2])
    @variable(model, s[1:N, 1:2])
    @variable(model, λ[1:N, 1:2] >= 0)

    @objective(model, Min, sum(Δt * 0.5 * (uA[k, :]' * uA[k, :]) + Δt * 0.5 * (uB[k, :]' * uB[k, :]) for k in 1:N-1))

    continuous_dynamics(x, u; m) = [x[3:4]; u ./ m]
    @constraint(model, [k=1:N-1], xA[k+1, :] .== xA[k, :] + Δt * continuous_dynamics(xA[k+1, :], uA[k, :]; m=m))
    @constraint(model, [k=1:N-1], xB[k+1, :] .== xB[k, :] + Δt * continuous_dynamics(xB[k+1, :], uB[k, :]; m=m))

    @constraints(model, begin
        xA[1, :] .== x0A
        xB[1, :] .== x0B
        xA[N, :] .== xfA
        xB[N, :] .== xfB
    end)

    @expression(model, dA[k=1:N], xA[k, 1:2] - xB[k, 1:2])

    @variable(model, w_plus[1:N, 1:2]  >= 0)   # = s[k,i] - dA[k][i]
    @variable(model, w_minus[1:N, 1:2] >= 0)   # = s[k,i] + dA[k][i]
    @variable(model, μ[1:N, 1:2]       >= 0)   # = 1 - λ[k,i]

    @constraint(model, [k=1:N, i=1:2], w_plus[k,i]  == s[k,i] - dA[k][i])
    @constraint(model, [k=1:N, i=1:2], w_minus[k,i] == s[k,i] + dA[k][i])
    @constraint(model, [k=1:N, i=1:2], μ[k,i]       == 1 - λ[k,i])

    for k in 1:N, i in 1:2
        @constraint(model, [λ[k,i], w_plus[k,i]]  in SOS1())
        @constraint(model, [μ[k,i], w_minus[k,i]]  in SOS1())
    end

    @constraint(model, [k=1:N], sum(s[k, :]) >= d_min)

    return model
end

##
using CCOpt

model = multi_robot_coordination2(N=30, T=5.0)

xA_inds = map(v -> JuMP.index(v).value, model[:xA])
xB_inds = map(v -> JuMP.index(v).value, model[:xB])
uA_inds = map(v -> JuMP.index(v).value, model[:uA])
uB_inds = map(v -> JuMP.index(v).value, model[:uB])
s_inds  = map(v -> JuMP.index(v).value, model[:s])
λ_inds      = vec(map(v -> JuMP.index(v).value, model[:λ]))
w_plus_inds = vec(map(v -> JuMP.index(v).value, model[:w_plus]))
w_minus_inds = vec(map(v -> JuMP.index(v).value, model[:w_minus]))
μ_inds       = vec(map(v -> JuMP.index(v).value, model[:μ]))

# nlp = MathOptNLPModel(model, name="Multi-Robot Coordination")
# ind_vcc1 = [λ_inds; μ_inds]
# ind_vcc2 = [w_plus_inds; w_minus_inds]

nlp, ind_vcc1, ind_vcc2 = to_mpcc(model)
mpcc = CCOpt.MPCCModelVarVar(nlp, ind_vcc1, ind_vcc2)
solver = CCOpt.RelaxationSolver(mpcc)
stats = CCOpt.solve_homotopy!(solver)

# mpcc = CCOpt.MPCCModelConCon(nlp, inds_ccc1, inds_ccc2)
# solver = CCOpt.RelaxationSolver(CCOpt.vertical_form(mpcc))
# stats = CCOpt.solve_homotopy!(solver)

##
using Gurobi

model = multi_robot_coordination2(N=30, T=5.0)

let N=30, λ = model[:λ], w_plus = model[:w_plus], w_minus = model[:w_minus], μ = model[:μ]
    for k in 1:N, i in 1:2
        @constraint(model, [λ[k,i], w_plus[k,i]]  in SOS1())
        @constraint(model, [μ[k,i], w_minus[k,i]]  in SOS1())
    end
        
    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
end

## Visualize the trajectories and L1 distance between the robots over time, similar to the animation in multi_robot_coordination.jl

using Plots, Printf; gr()

let N = 30, T = 5.0, d_min = 2.5
    ts = range(0.0, T, N)

    xA_val = value.(model[:xA])
    xB_val = value.(model[:xB])
    s_val  = value.(model[:s])

    rA = [xA_val[k, 1:2] for k in 1:N]
    vA = [xA_val[k, 3:4] for k in 1:N]
    rB = [xB_val[k, 1:2] for k in 1:N]
    vB = [xB_val[k, 3:4] for k in 1:N]
    l1_dist = [s_val[k, 1] + s_val[k, 2] for k in 1:N]

    all_x = [r[1] for r in [rA; rB]]
    all_y = [r[2] for r in [rA; rB]]
    pad = 1.5
    xlims_traj = (minimum(all_x) - pad, maximum(all_x) + pad)
    ylims_traj = (minimum(all_y) - pad, maximum(all_y) + pad)
    arrow_scale = 0.15

    p2 = plot(ts, l1_dist;
        label="‖rA − rB‖₁", color=:mediumpurple, lw=2.5, fill=(0, 0.08, :mediumpurple))
    hline!(p2, [d_min]; label="d_min = $(d_min)", color=:black, ls=:dash, lw=1.5)
    plot!(p2;
        xlabel="Time (s)", ylabel="L1 distance (m)", title="Robot Separation",
        legend=:topright, framestyle=:box, grid=true, gridalpha=0.3,
        ylims=(0, max(maximum(l1_dist) * 1.15, d_min * 2.0)))

    anim = @animate for k in 1:N
        p1 = plot(
            [r[1] for r in rA], [r[2] for r in rA]; color=:royalblue, lw=1, alpha=0.15, label="")
        plot!(p1,
            [r[1] for r in rB], [r[2] for r in rB]; color=:firebrick, lw=1, alpha=0.15, label="")

        if k > 1
            plot!(p1, [r[1] for r in rA[1:k]], [r[2] for r in rA[1:k]];
                label="Robot A", color=:royalblue, lw=2.5)
            plot!(p1, [r[1] for r in rB[1:k]], [r[2] for r in rB[1:k]];
                label="Robot B", color=:firebrick, lw=2.5)
        end

        scatter!(p1, [rA[1][1]], [rA[1][2]]; marker=:circle, ms=6, color=:royalblue, label="")
        scatter!(p1, [rB[1][1]], [rB[1][2]]; marker=:circle, ms=6, color=:firebrick,  label="")
        scatter!(p1, [rA[end][1]], [rA[end][2]]; marker=:rect, ms=6, color=:royalblue, label="", alpha=0.35)
        scatter!(p1, [rB[end][1]], [rB[end][2]]; marker=:rect, ms=6, color=:firebrick,  label="", alpha=0.35)
        scatter!(p1, [rA[k][1]], [rA[k][2]]; marker=:circle, ms=11, color=:royalblue, label="")
        scatter!(p1, [rB[k][1]], [rB[k][2]]; marker=:circle, ms=11, color=:firebrick,  label="")

        quiver!(p1, [rA[k][1]], [rA[k][2]];
            quiver=([vA[k][1] * arrow_scale], [vA[k][2] * arrow_scale]), color=:royalblue, lw=1.5)
        quiver!(p1, [rB[k][1]], [rB[k][2]];
            quiver=([vB[k][1] * arrow_scale], [vB[k][2] * arrow_scale]), color=:firebrick, lw=1.5)

        plot!(p1;
            xlabel="x (m)", ylabel="y (m)",
            title=@sprintf("Robot Trajectories  (t = %.2f s)", ts[k]),
            aspect_ratio=:equal, legend=:topleft, framestyle=:box,
            grid=true, gridalpha=0.3, xlims=xlims_traj, ylims=ylims_traj)

        plot(p1, p2; layout=(1, 2), size=(1000, 460), margin=5Plots.mm, dpi=120)
    end

    gif(anim; fps=15)
end


# Add SOS1 constraints for the complementarity conditions.
#
# The four KKT constraint containers encode two complementarity pairs per (k, i):
#   Pair 1:  0 ≤ λ[k,i]        ⟂  s[k,i] - dA[k][i] ≥ 0
#   Pair 2:  0 ≤ (1-λ[k,i])    ⟂  s[k,i] + dA[k][i] ≥ 0
#
# SOS1 requires variables, so we introduce slacks for the expressions that aren't
# already stand-alone variables: the two constraint RHS slacks and (1 - λ).
# let xA = model[:xA], xB = model[:xB], s = model[:s], λ = model[:λ],
#     dA = model[:dA], N = size(model[:xA], 1)

#     @variable(model, w_plus[1:N, 1:2]  >= 0)   # = s[k,i] - dA[k][i]
#     @variable(model, w_minus[1:N, 1:2] >= 0)   # = s[k,i] + dA[k][i]
#     @variable(model, μ[1:N, 1:2]       >= 0)   # = 1 - λ[k,i]

#     @constraint(model, [k=1:N, i=1:2], w_plus[k,i]  == s[k,i] - dA[k][i])
#     @constraint(model, [k=1:N, i=1:2], w_minus[k,i] == s[k,i] + dA[k][i])
#     @constraint(model, [k=1:N, i=1:2], μ[k,i]       == 1 - λ[k,i])

#     for k in 1:N, i in 1:2
#         @constraint(model, [λ[k,i],  w_plus[k,i]]  in SOS1())
#         @constraint(model, [μ[k,i], w_minus[k,i]]  in SOS1())
#     end
# end

# ##
# using CCOpt

# nlp, inds_ccc1, inds_ccc2 = multi_robot_coordination(N=30, T=5.0)

# mpcc = CCOpt.MPCCModelConCon(nlp, inds_ccc1, inds_ccc2)
# solver = CCOpt.RelaxationSolver(CCOpt.vertical_form(mpcc))
# stats = CCOpt.solve_homotopy!(solver)

# @register multi_robot_coordination :robotics

# # Instantiate probem
# problem = Marble.Problem(data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)

# # Create an instance of Marble and solve the problem
# solver = Marble.Solver()
# Marble.set_problem!(solver, problem)

# opts = Marble.SolverOptions()
# Marble.verbosity!(opts, true)

# result = Marble.solve!(solver, opts)

# ## Visualization
# using Plots, Printf; gr()

# N = 30; T = 5.0; d_min = 2.5
# ts = range(0.0, T, N)

# z = Marble.z(result)

# rA = [z[inds.xA[k][1:2]] for k = 1:N]
# vA = [z[inds.xA[k][3:4]] for k = 1:N]
# rB = [z[inds.xB[k][1:2]] for k = 1:N]
# vB = [z[inds.xB[k][3:4]] for k = 1:N]
# l1_dist = [sum(z[inds.s[k]]) for k = 1:N]

# # Fixed axis limits so the frame doesn't jump during animation
# all_x = [r[1] for r in [rA; rB]]
# all_y = [r[2] for r in [rA; rB]]
# pad = 1.5
# xlims_traj = (minimum(all_x) - pad, maximum(all_x) + pad)
# ylims_traj = (minimum(all_y) - pad, maximum(all_y) + pad)

# arrow_scale = 0.15

# # ── Static right plot (L1 norm) ───────────────────────────────────────────────
# p2 = plot(
#     ts, l1_dist;
#     label="‖rA − rB‖₁", color=:mediumpurple, lw=2.5, fill=(0, 0.08, :mediumpurple),
# )
# hline!(p2, [d_min];
#     label="d_min = $(d_min)", color=:black, ls=:dash, lw=1.5,
# )
# plot!(p2;
#     xlabel="Time (s)", ylabel="L1 distance (m)",
#     title="Robot Separation",
#     legend=:topright,
#     framestyle=:box,
#     grid=true, gridalpha=0.3,
#     ylims=(0, max(maximum(l1_dist) * 1.15, d_min * 2.0)),
# )

# # ── Animation ─────────────────────────────────────────────────────────────────
# anim = @animate for k = 1:N
#     # Ghost full-path trails
#     p1 = plot(
#         [r[1] for r in rA], [r[2] for r in rA];
#         color=:royalblue, lw=1, alpha=0.15, label="",
#     )
#     plot!(p1,
#         [r[1] for r in rB], [r[2] for r in rB];
#         color=:firebrick, lw=1, alpha=0.15, label="",
#     )

#     # Trace so far
#     if k > 1
#         plot!(p1,
#             [r[1] for r in rA[1:k]], [r[2] for r in rA[1:k]];
#             label="Robot A", color=:royalblue, lw=2.5,
#         )
#         plot!(p1,
#             [r[1] for r in rB[1:k]], [r[2] for r in rB[1:k]];
#             label="Robot B", color=:firebrick, lw=2.5,
#         )
#     end

#     # Start markers (always visible)
#     scatter!(p1, [rA[1][1]], [rA[1][2]]; marker=:circle, ms=6, color=:royalblue, label="")
#     scatter!(p1, [rB[1][1]], [rB[1][2]]; marker=:circle, ms=6, color=:firebrick,  label="")

#     # Goal markers (always visible, hollow)
#     scatter!(p1, [rA[end][1]], [rA[end][2]]; marker=:rect, ms=6, color=:royalblue, label="", alpha=0.35)
#     scatter!(p1, [rB[end][1]], [rB[end][2]]; marker=:rect, ms=6, color=:firebrick,  label="", alpha=0.35)

#     # Current position dots
#     scatter!(p1, [rA[k][1]], [rA[k][2]]; marker=:circle, ms=11, color=:royalblue, label="")
#     scatter!(p1, [rB[k][1]], [rB[k][2]]; marker=:circle, ms=11, color=:firebrick,  label="")

#     # Velocity arrows at current position
#     quiver!(p1, [rA[k][1]], [rA[k][2]];
#         quiver=([vA[k][1] * arrow_scale], [vA[k][2] * arrow_scale]),
#         color=:royalblue, lw=1.5,
#     )
#     quiver!(p1, [rB[k][1]], [rB[k][2]];
#         quiver=([vB[k][1] * arrow_scale], [vB[k][2] * arrow_scale]),
#         color=:firebrick, lw=1.5,
#     )

#     plot!(p1;
#         xlabel="x (m)", ylabel="y (m)",
#         title=@sprintf("Robot Trajectories  (t = %.2f s)", ts[k]),
#         aspect_ratio=:equal,
#         legend=:topleft,
#         framestyle=:box,
#         grid=true, gridalpha=0.3,
#         xlims=xlims_traj, ylims=ylims_traj,
#     )

#     plot(p1, p2; layout=(1, 2), size=(1000, 460), margin=5Plots.mm, dpi=120)
# end

# gif(anim; fps=15)
