using LinearAlgebra

function mpcc_licq(data::NamedTuple, x::AbstractVector; rtol=1e-6, τ=1e-7, verbose=true)
    (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = data

    G = L * x .+ l
    H = R * x .+ r
    g = J_ineq * x + b_ineq
    m = length(G)

    # Classify each complementarity pair G ⟂ H by the rel magnitude of its
    # two sides: whichever side is at least `rtol` times
    # smaller than the other is the active/tight one. If the two sides are within
    # `rtol` of each other (e.g. both ~0), neither clearly vanishes and the pair is
    # biactive
    IG  = findall(i -> abs(G[i]) < rtol * abs(H[i]), 1:m)   # G << H  -> L active
    IH  = findall(i -> abs(H[i]) < rtol * abs(G[i]), 1:m)   # H << G  -> R active
    I00 = setdiff(1:m, vcat(IG, IH))                        # comparable -> biactive

    # Active inequalities use an absolute tolerance
    Ig  = findall(j -> abs(g[j]) <= τ, eachindex(g))

    J = vcat(J_eq, J_ineq[Ig, :], L[vcat(IG, I00), :], R[vcat(IH, I00), :])

    s = svdvals(J)
    σmin = isempty(s) ? Inf : minimum(s)
    σmax = isempty(s) ? 0.0 : maximum(s)
    licq = size(J, 1) ≤ size(J, 2) && (isempty(s) || σmin > 1e-8 * σmax)

    if verbose
        @info "MPCC LICQ: $(licq ? "satisfied" : "violated")" size_J=size(J) σ_min=σmin σ_max=σmax L_active=length(IG) R_active=length(IH) biactive=length(I00)
    end

    return licq
end
