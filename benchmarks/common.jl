using AmplNLReader
using DataFrames
using Dates
using CSV
using MadNLP

madnlp_status_to_string = Dict(s => string(s) for s in instances(MadNLP.Status))

# function add_ccopt_stats!(df::DataFrame, name::String, stats)
#     push!(df, (
#         problem=name,
#         status=stats.status,
#         obj=stats.objective,
#         total_time_s=stats.counters.total_time,
#     ))
# end