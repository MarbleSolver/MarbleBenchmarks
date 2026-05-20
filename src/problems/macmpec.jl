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
    ind_ccc2 = findall(!iszero, nlp.meta.cvar)
    ind_vcc1 = nlp.meta.cvar[ind_ccc2]
    cc_types = (:var, :con)

    return (; model=nlp, ind_cc1=ind_vcc1, ind_cc2=ind_ccc2, cc_types=cc_types, params=(; name))
end

for name in MACMPEC_QPCCs
    register(() -> macmpec(name=name), Symbol(name), :macmpec)
end