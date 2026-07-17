module MarbleBenchmarks

using LinearAlgebra, SparseArrays
using NLPModels, AmplNLReader, JuMP, NLPModelsJuMP
using Marble
using BlockDiagonals, Combinatorics, ControlSystems

include("quaternion_helpers.jl")

include("reformulation.jl")
export mpcc_to_sos1

include("mpcc_licq.jl")
export mpcc_licq

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
export quadrotor_gates, three_d_gates, three_d_gates_fewer_comps, multi_robot_coordination,
       multi_robot_obstacle_avoidance, rocket_catch, block_push_1d, hopper_2d,
       transport, waiter

include("problems/quadrotor_gates_3d.jl")
include("problems/multi_robot_coordination.jl")
include("problems/multi_robot_obstacle_avoidance.jl")
include("problems/rocket_catch_3d.jl")
include("problems/rocket_catch_2d.jl")
include("problems/block_push_1d.jl")
include("problems/hopper_2d.jl")
include("problems/transport.jl")
include("problems/waiter.jl")
include("problems/macmpec.jl")
include("problems/lane_passing_2d.jl")

# Benchmark harness as a nested submodule. NOTE: this pulls in the solver
# backends (Gurobi, CRISP, LCQPow), so `using MarbleBenchmarks` loads them too.
# Run the benchmarks via `using MarbleBenchmarks.Benchmarks`.
module Benchmarks

using LinearAlgebra
using DataFrames, CSV, Dates, Serialization
using JuMP
using Marble
using Gurobi
using CRISP, LCQPow
using ..MarbleBenchmarks: get_problem, mpcc_to_sos1
const MOI = JuMP.MOI

include("../benchmarks/common.jl")
include("../benchmarks/solvers/marble.jl")
include("../benchmarks/solvers/gurobi.jl")
include("../benchmarks/solvers/crisp.jl")
include("../benchmarks/solvers/lcqpow.jl")

export marble_solve, gurobi_solve, crisp_solve, lcqpow_solve
export run_benchmark, BenchmarkProblemResult, get_mpcc, feasibility_stats

end # module Benchmarks

export Benchmarks

end
