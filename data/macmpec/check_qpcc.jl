using AmplNLReader, NLPModels, LinearAlgebra, JSON

const NL_DIR    = joinpath(@__DIR__, "MacMPEC-nl")
const JSON_PATH = joinpath(@__DIR__, "macmpec_qpcc.json")
const TOL = 1e-8

data = JSON.parsefile(JSON_PATH)

println("Checking $(length(data)) MacMPEC problems...\n")
n_true_qpcc = 0

for (i, (name, entry)) in enumerate(data)
    global n_true_qpcc
    
    # safe_name = replace(name, r"[^a-zA-Z0-9_\-.]" => "_")
    nl_path   = joinpath(NL_DIR, name * ".nl")
    isfile(nl_path) || (println("  SKIP      $name  (no .nl file)"); continue)

    classification = split(entry["classification"], "-")
    cost_class = classification[1][1]
    cons_class = classification[1][2]
    smooth_class = classification[1][3]

    try
        nlp = AmplModel(nl_path)

        # Constraint linearity: meta.nnln counts nonlinear general constraints.
        # Bound and complementarity constraints are not counted here.
        all_linear = nlp.meta.nnln == 0
        entry["linear_constraints"] = all_linear

        if cost_class ∉ ['N', 'C', 'L', 'Q', 'S']
            is_convex = false
        else
            # Objective convexity: Hessian is constant for quadratic objectives.
            # Flip sign for maximization so PSD check is uniform.
            sign      = nlp.meta.minimize ? 1 : -1
            H         = hess(nlp, nlp.meta.x0; obj_weight = 1.0)
            λ_min     = minimum(eigvals(Symmetric(sign .* Matrix(H))))
            is_convex = λ_min >= -TOL
            entry["convex_cost"] = is_convex
        end

        sense    = nlp.meta.minimize ? "min" : "max"
        
        # Update count
        all_linear && is_convex && (n_true_qpcc += 1)

        lin_tag  = all_linear ? "linear   " : "NONLINEAR"
        conv_tag = is_convex  ? "convex  " : "NONCONVEX"
        println("  $i $lin_tag  $conv_tag  [$sense]  $name  " *
                "(nlin=$(nlp.meta.nlin), nnln=$(nlp.meta.nnln))")
    catch e
        entry["linear_constraints"] = false
        entry["convex_cost"]   = false
        println("  $i SKIP      $name  (failed: $e)")
    end
end

open(JSON_PATH, "w") do io
    JSON.print(io, data, 2)
end
println("\nUpdated $JSON_PATH")
println("  $n_true_qpcc / $(length(data)) problems are truly QPCC (linear constraints + convex cost)")
