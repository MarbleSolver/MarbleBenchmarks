module MarbleBenchmarks

using LinearAlgebra, SparseArrays
using NLPModels, AmplNLReader, JuMP, NLPModelsJuMP
using Marble
using BlockDiagonals, Combinatorics, ControlSystems

include("jump_mpcc.jl")
# include("julia_marble_solver.jl")

export nlp_con_row_map, nlp_var_col_map, var_var_complementarities, var_con_complementarities, con_con_complementarities
export var_inds, reformulate_sos1, reformulate_lie
# export JuliaMarbleOptions, JuliaMarbleResult, solve_julia_marble

# using MPCCModels

const PROBLEM_REGISTRY = Dict{Symbol, Tuple{Function, Symbol}}()

register(fn::Function, class::Symbol = :general) = (PROBLEM_REGISTRY[Symbol(fn)] = (fn, class))

function get_problem(name::Symbol; kwargs...)
    haskey(PROBLEM_REGISTRY, name) || error("Unknown problem: $name. Available: $(list_problems())")
    return PROBLEM_REGISTRY[name][1](; kwargs...)
end

list_problems() = sort(collect(keys(PROBLEM_REGISTRY)))
list_problems(class::Symbol) = sort([k for (k, (_, c)) in PROBLEM_REGISTRY if c == class])
problem_classes() = sort(unique(c for (_, c) in values(PROBLEM_REGISTRY)))

export get_problem, list_problems, problem_classes, register
export quadrotor_gates, quadrotor_gates_double_integrator, multi_robot_coordination

include("problems/quadrotor_gates.jl")
include("problems/quadrotor_gates_double_integrator.jl")
include("problems/multi_robot_coordination.jl")

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

end
