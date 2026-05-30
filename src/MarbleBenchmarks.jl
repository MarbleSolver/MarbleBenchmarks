module MarbleBenchmarks

using LinearAlgebra, SparseArrays
using NLPModels, AmplNLReader, JuMP, NLPModelsJuMP
using Marble
using BlockDiagonals, Combinatorics, ControlSystems

include("jump_mpcc.jl")
include("quaternion_helpers.jl")
# include("julia_marble_solver.jl")

export nlp_con_row_map, nlp_var_col_map, var_var_complementarities, var_con_complementarities, con_con_complementarities
export var_inds, reformulate_sos1, reformulate_lie
# export JuliaMarbleOptions, JuliaMarbleResult, solve_julia_marble

# using MPCCModels

const PROBLEM_REGISTRY = Dict{Symbol, Tuple{Function, Symbol}}()

register(fn::Function, class::Symbol = :general) = (PROBLEM_REGISTRY[Symbol(fn)] = (fn, class))
register(fn::Function, name, class::Symbol = :general) = (PROBLEM_REGISTRY[Symbol(name)] = (fn, class))

function get_problem(name::Symbol; kwargs...)
    haskey(PROBLEM_REGISTRY, name) || error("Unknown problem: $name. Available: $(list_problems())")
    return PROBLEM_REGISTRY[name][1](; kwargs...)
end

list_problems() = sort(collect(keys(PROBLEM_REGISTRY)))
list_problems(class::Symbol) = sort([k for (k, (_, c)) in PROBLEM_REGISTRY if c == class])
problem_classes() = sort(unique(c for (_, c) in values(PROBLEM_REGISTRY)))

export get_problem, list_problems, problem_classes, register
export quadrotor_gates, multi_robot_coordination, rocket_catch, block_pushing_1d

include("problems/quadrotor_gates.jl")
include("problems/multi_robot_coordination.jl")
include("problems/rocket_catch.jl")
include("problems/block_pushing_1d.jl")
include("problems/macmpec.jl")

end
