module MarbleBenchmarks

using LinearAlgebra, SparseArrays
using NLPModels, AmplNLReader, JuMP, NLPModelsJuMP
using Marble
using BlockDiagonals, Combinatorics, ControlSystems

include("quaternion_helpers.jl")

include("reformulation.jl")
export mpcc_to_sos1

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
export quadrotor_gates, three_d_gates, three_d_gates_fewer_comps, multi_robot_coordination, rocket_catch, block_push_1d

include("problems/quadrotor_gates.jl")
include("problems/3d_gates.jl")
include("problems/multi_robot_coordination.jl")
include("problems/rocket_catch_3d.jl")
include("problems/block_push_1d.jl")
include("problems/macmpec.jl")
include("problems/minimal_gate_mpcc.jl")
include("problems/minimal_gate_mpcc_cc.jl")

end
