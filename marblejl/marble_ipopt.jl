using Marble, MarbleBenchmarks
using Ipopt
using JuMP, NLPModelsJuMP
using NLPModels

@inline function bκ(t; κ)
    return 0.5 * (t + √(t^2 + 4κ))
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

    return (;
        Q, q, c0,
        A, b,
        G, h,
        L, l,
        R, r,
        nvar, neq, nineq, ncc,
    )
end

function test_solve(data; κ)
    (; Q, q, c0, A, b, G, h, L, l, R, r, nvar, neq, nineq, ncc) = data

    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:nvar])
    @variable(model, s[1:ncc])

    @objective(model, Min, 0.5 * x' * Q * x + q' * x + c0)

    @constraint(model, A * x + b .== 0)
    @constraint(model, G * x + h .>= 0)
    @constraint(model, L * x + l .== bκ.(s; κ=κ))
    @constraint(model, R * x + r .== bκ.(-s; κ=κ))

    optimize!(model)
    xstar = value.(x)
    sstar = value.(s)

    return (; x=xstar, s=sstar)
end

function macmpec(name)
    problem = get_problem(name)
    nlp = problem.model isa AbstractNLPModel ? problem.model : MathOptNLPModel(problem.model)
    Q, q, c0, A, b, G, h, L, l, R, r = jump_to_marble(nlp, problem.ind_cc1, problem.ind_cc2, problem.cc_types)
    data = make_data(Q, q, c0, A, b, G, h, L, l, R, r)

    return data
end

##

data = macmpec(Symbol("multi_robot_coordination"))
result = test_solve(data; κ=1e-4)
x = result.x
s = result.s

obj = 0.5 * x' * data.Q * x + data.q' * x + data.c0
comp = norm((data.L * x + data.l) .* (data.R * x + data.r), Inf)

@info "Solution stats" obj comp