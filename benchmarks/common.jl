using AmplNLReader
using DataFrames
using Dates
using CSV
using MadNLP

madnlp_status_to_string = Dict(s => string(s) for s in instances(MadNLP.Status))