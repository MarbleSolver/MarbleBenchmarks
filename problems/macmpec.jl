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
