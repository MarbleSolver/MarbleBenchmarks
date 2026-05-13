module MarbleBenchmarks

using LinearAlgebra, SparseArrays
using NLPModels, ADNLPModels, AmplNLReader, JuMP
using Marble
using MPCCModels

# include("nlp_utils.jl")
# export CBuilder, add!, add_eq!, add_ineq!, add_cc!, assemble

# include("indexed_components.jl")
# export indexed_components

struct ProblemEntry
    fn::Function
    type::Symbol
end

const PROBLEM_REGISTRY = Dict{Symbol, ProblemEntry}()

function get_problem(name::Symbol; kwargs...)
    haskey(PROBLEM_REGISTRY, name) || error(
        "Unknown problem: $name. Available: $(list_problems())"
    )
    return PROBLEM_REGISTRY[name].fn(; kwargs...)
end

# @register fn                  — named fn, type = :general
# @register fn :type            — named fn, explicit type
# @register sym_expr fn_expr    — dynamic name, type = :general
# @register sym_expr fn_expr :type — dynamic name, explicit type
macro register(fn)
    sym = QuoteNode(fn)
    quote
        export $(esc(fn))
        PROBLEM_REGISTRY[$sym] = ProblemEntry($(esc(fn)), :general)
    end
end

macro register(a, b)
    if a isa Symbol  # @register fn :type
        sym = QuoteNode(a)
        quote
            export $(esc(a))
            PROBLEM_REGISTRY[$sym] = ProblemEntry($(esc(a)), $(esc(b)))
        end
    else             # @register sym_expr fn_expr
        quote
            PROBLEM_REGISTRY[$(esc(a))] = ProblemEntry($(esc(b)), :general)
        end
    end
end

macro register(sym_expr, fn_expr, problem_type)
    quote
        PROBLEM_REGISTRY[$(esc(sym_expr))] = ProblemEntry($(esc(fn_expr)), $(esc(problem_type)))
    end
end

list_problems() = sort(collect(keys(PROBLEM_REGISTRY)))
list_problems(type::Symbol) = sort([k for (k, v) in PROBLEM_REGISTRY if v.type == type])
problem_types() = sort(unique(v.type for v in values(PROBLEM_REGISTRY)))

export get_problem, list_problems, problem_types, @register

# for _f in readdir(joinpath(@__DIR__, "..", "problems"), join=true)
#     endswith(_f, ".jl") && include(_f)
# end

include("conversion.jl")
export to_mpcc, to_jump

end
