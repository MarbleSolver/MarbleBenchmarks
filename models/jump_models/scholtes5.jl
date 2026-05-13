# scholtes5.mod	QBR-AN-LCP-3-2-2
# Original AMPL coding by Sven Leyffer

# A QPEC from S. Scholtes, Judge Inst., University of Cambridge.
# see S. Scholtes, "Convergence properties of a regularization
# scheme for MPCCs", SIAM J. Optimization 11(4):918-936, 2001.

using JuMP

function build_model()
    model = Model()

    @variable(model, z[1:3] >= 0)

    @objective(model, Min, (z[1] - 1)^2 + (z[2] - 2)^2 + (z[3] + 1)^2)

    @constraint(model, compl1, [z[1], z[3]] in SOS1())
    @constraint(model, compl2, [z[2], z[3]] in SOS1())

    return model
end

model = build_model()
