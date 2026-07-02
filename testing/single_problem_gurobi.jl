using MarbleBenchmarks
using Gurobi
using JuMP

const GRB_ENV = redirect_stdout(devnull) do
    Gurobi.Env()
end

## 

# problem = get_problem(:rocket_catch_3d; N=40, T=4.1);
problem = get_problem(:quadrotor_gates_3d; N=30, T=9.0, ngates=4);

model = mpcc_to_sos1(problem.mpcc)

set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
# for (k, v) in pairs(GUROBI_OPTIONS)
#     set_attribute(model, String(k), v)
# end
optimize!(model)
solve_time(model) * 1e3
## sl1 problem in JuMP

model = JuMP.Model()
@variable(model, x[1:2])


name = Symbol("sl1")
problem = MarbleBenchmarks.get_problem(name)

gurobi_model = MarbleBenchmarks.reformulate_sos1(problem.model, problem.ind_cc1, problem.ind_cc2, problem.cc_types)
set_optimizer(gurobi_model, Gurobi.Optimizer)
optimize!(gurobi_model)

status = termination_status(gurobi_model)
primal  = primal_status(gurobi_model)

println("Gurobi status: $(status)")
println("Gurobi primal status: $(primal)")
println("Gurobi solve time: $(solve_time(gurobi_model))")

if primal == FEASIBLE_POINT
    z_gurobi = value.(all_variables(gurobi_model))[1:num_variables(problem.model)]
else
    error("Gurobi did not find a feasible solution.")
end

