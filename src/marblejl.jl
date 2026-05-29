using Marble

function solve(data::MarbleData)
    # Solves the following QPCC problem:
    # min  1/2 * x'*H*x + g'*x + f0
    # s.t. J_eq * x + b_eq = 0
    #      J_ineq * x + b_ineq ≥ 0
    #      0 ≤ L * x + l ⟂ R * x + r ≥ 0
    # 
    # Relaxed to:
    # min  1/2 * x'*H*x + g'*x + f0
    # s.t.  J_eq * x + b_eq = 0
    #       J_ineq * x + b_ineq ≥ 0
    #       L * x + l = exp(s)
    #       R * x + r = exp(t)
    #       s + t ≤ ln(κ)

    # for κ ∈ [1, 0.1, ..., κ_min]:
    #     solve the relaxed problem using Permenter's method for inequalities
end