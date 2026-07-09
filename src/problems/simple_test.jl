function simple_test()
    # Construct problem using JuMP
    model = JuMP.Model()

    @variable(model, x[1:4])
    @objective(model, Min, x'*x)
    @constraint(model, x[1] == 1)
    @constraint(model, x[2] >= 1)

    # For complementarities, we define each inequality and then specify indices
    # for the complementarity pairs
    @constraint(model, x3_comp, x[3] + 1 >= 0)
    @constraint(model, x4_comp, x[4] - 1 >= 0)

    return MPCC(model,
        (model[:x3_comp], model[:x4_comp])
    )
end

register(simple_test)
