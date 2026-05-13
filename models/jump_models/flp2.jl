# flp2.mod	QLR-AN-LCP-4-2-2
#
# Problem 2 from Fukushima, M. Luo, Z.-Q.Pang, J.-S.,
# "A globally convergent Sequential Quadratic Programming
# Algorithm for Mathematical Programs with Linear Complementarity
# Constraints", Computational Optimization and Applications, 10(1),
# pp. 5-34, 1998.
#
# Note: Problem 1 is equivalent to stackelberg1.mod

using JuMP

function build_model()
    model = Model()

    @variable(model, 0 <= x[1:2] <= 10)
    @variable(model, y[1:2] >= 0)

    @objective(model, Min, 0.5*( (x[1]+x[2]+y[1]-15)^2 + (x[1]+x[2]+y[2]-15)^2 ))

    @variable(model, compl1_s >= 0)
    @constraint(model, compl1_def, compl1_s == 8/3*x[1] + 2*x[2] + 2*y[1] + 8/3*y[2] - 36)
    @constraint(model, compl1, [y[1], compl1_s] in SOS1())

    @variable(model, compl2_s >= 0)
    @constraint(model, compl2_def, compl2_s == 2*x[1] + 5/4*x[2] + 5/4*y[1] + 2*y[2] - 25)
    @constraint(model, compl2, [y[2], compl2_s] in SOS1())

    return model
end

model = build_model()
