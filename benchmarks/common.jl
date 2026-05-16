using AmplNLReader
using DataFrames
using Dates
using CSV

nl_path(name) = begin
    name ∈ MarbleBenchmarks.MACMPEC_QPCCs || @warn "Problem $name not found in MacMPEC QPCC collection"

    joinpath(@__DIR__, "..", "data", "macmpec_nls", "$name.nl")
end

macmpec_to_mpcc(name) = begin    
    nlp = AmplModel(nl_path(name))
    ind_ccc2 = findall(!iszero, nlp.meta.cvar)
    ind_vcc1 = nlp.meta.cvar[ind_ccc2]

    return nlp, ind_vcc1, ind_ccc2
end

madnlp_status_to_string = Dict(s => string(s) for s in instances(MadNLP.Status))

# function add_ccopt_stats!(df::DataFrame, name::String, stats)
#     push!(df, (
#         problem=name,
#         status=stats.status,
#         obj=stats.objective,
#         total_time_s=stats.counters.total_time,
#     ))
# end