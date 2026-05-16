using AmplNLReader
using JSON

const _macmpec_problem_names = keys(JSON.parsefile(joinpath(@__DIR__, "..", "data", "macmpec", "macmpec_lcqp.json"))) |> collect

function _macmpec(name::String)
    nl_path = joinpath(@__DIR__, "..", "data", "macmpec", "ampl_nl", name * ".nl")
    ampl = AmplModel(nl_path)
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MPCCModels.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

for _name in _macmpec_problem_names
    _sym = Symbol(replace(_name, r"[^a-zA-Z0-9_]" => "_"))
    @eval begin
        function $(_sym)(; kwargs...)
            _macmpec($(_name))
        end
        @register $(_sym) :macmpec
    end
end

##

using QuadraticModels
using SparseArrays
using LinearAlgebra
using MPCCModels
using ADNLPModels, NLPModels

H = Float64.(I(2))
g = zeros(2)

A = Float64[1 1; 1 0; 0 -1]
lcon = Float64[10, 0.0, -1.0]
ucon = Float64[Inf, Inf, Inf]

lvar = -Inf*ones(2)
uvar = Inf*ones(2)

f(x) = 0.5 * x' * H * x + g' * x
c(x) = A * x

nlp = ADNLPModel(f, zeros(2), lvar, uvar, c, lcon, ucon)

ind_ccc1 = [2]
ind_ccc2 = [3]

mpcc = MPCCModels.MPCCModelConCon(nlp, ind_ccc1, ind_ccc2)
vmpcc = MPCCModels.vertical_form(mpcc)

lifted_nlp = vmpcc.nlp   # a LiftedNLPModel wrapping the original

# Complementarity pair indices (both now refer to variables, not constraints)
ind1 = MPCCModels.get_ind_cc1(vmpcc)   # left-side variable indices
ind2 = MPCCModels.get_ind_cc2(vmpcc)   # right-side variable indices
ncc = MPCCModels.get_ncc(vmpcc)
MPCCModels.get_cc_types(vmpcc)   # all VarVar
ind_c = MPCCModels.get_ind_c(vmpcc)
n_orig = lifted_nlp.nlp.meta.nvar          # original variables
n_total = lifted_nlp.meta.nvar             # original + slacks
ind_slacks = MPCCModels.get_ind_lift_var(lifted_nlp.meta)  # which var indices are slacks
