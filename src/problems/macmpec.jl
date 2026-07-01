const MACMPEC_QPCCs = [
    "bard1",
    "bard1m",
    "bard2",
    "bilevel2",
    "bilevel2m",
    "ex9.2.1",
    "ex9.2.2",
    "ex9.2.4",
    "ex9.2.5",
    "ex9.2.6",
    "ex9.2.7",
    "flp2",
    "flp4-1",
    "flp4-2",
    "flp4-3",
    "flp4-4",
    "gauvin",
    "hs044-i",
    "jr1",
    "jr2",
    "kth2",
    "kth3",
    "liswet1-050",
    "liswet1-100",
    "liswet1-200",
    "nash1a",
    "nash1b",
    "nash1c",
    "nash1d",
    "nash1e",
    "portfl1",
    "portfl2",
    "portfl3",
    "portfl4",
    "portfl6",
    "qpec1",
    "qpec2",
    "scholtes3",
    "sl1",
]

nl_path(name) = joinpath(@__DIR__, "..", "..", "data", "macmpec_nls", "$name.nl")

function macmpec(; name)
    nlp = AmplModel(nl_path(name))

    # AMPL marks each complementarity constraint row with the index of the
    # variable it is paired with (0 otherwise), so each pair is `var ⊥ con`.
    ind_con = findall(!iszero, nlp.meta.cvar)   # complementarity constraint rows
    ind_var = nlp.meta.cvar[ind_con]            # variable each row is paired with
    cc_types = fill((:var, :con), length(ind_var))

    return MPCC(nlp, ind_var, ind_con, cc_types)
end

for name in MACMPEC_QPCCs
    register(() -> macmpec(name=name), Symbol(name), :macmpec)
end