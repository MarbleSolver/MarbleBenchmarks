using JuMP


"""
    nlp_con_row_map(model) -> Dict{MOI.ConstraintIndex, Int}

Maps each non-variable-bound constraint in `model` to its row index in the
NLPModel produced by `MathOptNLPModel(model)`. Iterates constraint types in the
same order MOI/NLPModelsJuMP does, skipping `VariableIndex` constraints (those
become variable bounds, not NLP rows).
"""
function nlp_con_row_map(model::JuMP.Model)
    moi_model = JuMP.backend(model)
    row = 0
    map = Dict{MOI.ConstraintIndex, Int}()
    for (F, S) in MOI.get(moi_model, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        for ci in MOI.get(moi_model, MOI.ListOfConstraintIndices{F,S}())
            row += 1
            map[ci] = row
        end
    end
    return map
end

"""
    nlp_var_col_map(model) -> Dict{MOI.VariableIndex, Int}

Maps each variable to its column index in the NLPModel produced by
`MathOptNLPModel(model)`.
"""
function nlp_var_col_map(model::JuMP.Model)
    vars = MOI.get(JuMP.backend(model), MOI.ListOfVariableIndices())
    return Dict(vi => i for (i, vi) in enumerate(vars))
end

mutable struct ComplementarityIndexBuilder
    model::JuMP.Model
    col_map::Dict{MOI.VariableIndex, Int}
    row_map::Dict{MOI.ConstraintIndex, Int}
    ind_cc1::Vector{Int}
    ind_cc2::Vector{Int}
    cc_types::Vector{Tuple{Symbol, Symbol}}
end

"""
    ComplementarityIndexBuilder(model)

Create a cached complementarity-index builder for a JuMP model. The builder
computes the NLP variable-column and constraint-row maps once, then reuses them
while adding any number of complementarity blocks. Create it after the variables
and constraints you want to index have been added to the model.
"""
function ComplementarityIndexBuilder(model::JuMP.Model)
    return ComplementarityIndexBuilder(
        model,
        nlp_var_col_map(model),
        nlp_con_row_map(model),
        Int[],
        Int[],
        Tuple{Symbol, Symbol}[],
    )
end

_assert_same_shape(a, b) = @assert size(a) == size(b) "Complementarity arguments must have the same shape"

_var_indices(builder::ComplementarityIndexBuilder, vars::AbstractArray) =
    [builder.col_map[JuMP.index(v)] for v in vec(vars)]

_con_indices(builder::ComplementarityIndexBuilder, cons::AbstractArray) =
    [builder.row_map[JuMP.index(c)] for c in vec(cons)]

function _append_complementarity_block!(
        builder::ComplementarityIndexBuilder,
        inds1,
        inds2,
        cc_type::Tuple{Symbol, Symbol},
    )
    append!(builder.ind_cc1, inds1)
    append!(builder.ind_cc2, inds2)
    append!(builder.cc_types, fill(cc_type, length(inds1)))
    return builder
end

function add_var_var_complementarities!(
        builder::ComplementarityIndexBuilder,
        vars1::AbstractArray,
        vars2::AbstractArray,
    )
    _assert_same_shape(vars1, vars2)
    return _append_complementarity_block!(
        builder,
        _var_indices(builder, vars1),
        _var_indices(builder, vars2),
        (:var, :var),
    )
end

function add_var_con_complementarities!(
        builder::ComplementarityIndexBuilder,
        vars::AbstractArray,
        cons::AbstractArray,
    )
    _assert_same_shape(vars, cons)
    return _append_complementarity_block!(
        builder,
        _var_indices(builder, vars),
        _con_indices(builder, cons),
        (:var, :con),
    )
end

function add_con_var_complementarities!(
        builder::ComplementarityIndexBuilder,
        cons::AbstractArray,
        vars::AbstractArray,
    )
    _assert_same_shape(cons, vars)
    return _append_complementarity_block!(
        builder,
        _con_indices(builder, cons),
        _var_indices(builder, vars),
        (:con, :var),
    )
end

function add_con_con_complementarities!(
        builder::ComplementarityIndexBuilder,
        cons1::AbstractArray,
        cons2::AbstractArray,
    )
    _assert_same_shape(cons1, cons2)
    return _append_complementarity_block!(
        builder,
        _con_indices(builder, cons1),
        _con_indices(builder, cons2),
        (:con, :con),
    )
end

complementarity_indices(builder::ComplementarityIndexBuilder) =
    builder.ind_cc1, builder.ind_cc2, builder.cc_types

"""
    complementarity_indices(model, blocks...)

Build complementarity indices for many blocks while computing the NLP index maps
only once. Each block is a tuple `(kind, lhs, rhs)`, where `kind` is one of
`:var_var`, `:var_con`, `:con_var`, or `:con_con`.

Example:

```julia
ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
    (:var_con, model[:λ], model[:gap_nonnegative]),
    (:var_var, model[:a], model[:b]),
)
```
"""
function complementarity_indices(model::JuMP.Model, blocks...)
    builder = ComplementarityIndexBuilder(model)
    for block in blocks
        @assert length(block) == 3 "Each complementarity block must be `(kind, lhs, rhs)`"
        kind, lhs, rhs = block
        if kind == :var_var
            add_var_var_complementarities!(builder, lhs, rhs)
        elseif kind == :var_con
            add_var_con_complementarities!(builder, lhs, rhs)
        elseif kind == :con_var
            add_con_var_complementarities!(builder, lhs, rhs)
        elseif kind == :con_con
            add_con_con_complementarities!(builder, lhs, rhs)
        else
            error("Unsupported complementarity block kind: $kind")
        end
    end
    return complementarity_indices(builder)
end

"""
    var_con_complementarities(model, vars, cons) -> Vector{Tuple{Int,Int}}

Given parallel arrays `vars` (VariableRef) and `cons` (ConstraintRef) with the
same shape, return `(col_idx, row_idx)` pairs suitable as complementarity indices
in an NLPModel.
"""
function var_con_complementarities(model::JuMP.Model, vars::AbstractArray, cons::AbstractArray)
    return var_con_complementarities(ComplementarityIndexBuilder(model), vars, cons)
end

function var_con_complementarities(builder::ComplementarityIndexBuilder, vars::AbstractArray, cons::AbstractArray)
    _assert_same_shape(vars, cons)
    return collect(zip(_var_indices(builder, vars), _con_indices(builder, cons)))
end

"""
    con_con_complementarities(model, cons1, cons2) -> Vector{Tuple{Int,Int}}

Given parallel arrays `cons1` and `cons2` (ConstraintRef) with the same shape,
return `(row_idx1, row_idx2)` pairs suitable as complementarity indices in an
NLPModel.
"""
function con_con_complementarities(model::JuMP.Model, cons1::AbstractArray, cons2::AbstractArray)
    return con_con_complementarities(ComplementarityIndexBuilder(model), cons1, cons2)
end

function con_con_complementarities(builder::ComplementarityIndexBuilder, cons1::AbstractArray, cons2::AbstractArray)
    _assert_same_shape(cons1, cons2)
    return collect(zip(_con_indices(builder, cons1), _con_indices(builder, cons2)))
end

"""
    var_var_complementarities(model, vars1, vars2) -> Vector{Tuple{Int,Int}}

Given parallel arrays `vars1` and `vars2` (VariableRef) with the same shape,
return `(col_idx1, col_idx2)` pairs suitable as complementarity indices in an
NLPModel. Both variables must be bounded below by zero.
"""
function var_var_complementarities(model::JuMP.Model, vars1::AbstractArray, vars2::AbstractArray)
    return var_var_complementarities(ComplementarityIndexBuilder(model), vars1, vars2)
end

function var_var_complementarities(builder::ComplementarityIndexBuilder, vars1::AbstractArray, vars2::AbstractArray)
    _assert_same_shape(vars1, vars2)
    return collect(zip(_var_indices(builder, vars1), _var_indices(builder, vars2)))
end

"""
    var_inds(model) -> Dict{Symbol, Array{Int}}

Returns a mapping from each named variable array in `model` (registered via
`@variable`) to an array of the corresponding MOI column indices (1-based).
Only entries whose value is an `AbstractArray{VariableRef}` are included;
scalar variables and non-variable objects are skipped.

Useful for building index lookups before constructing an NLPModel, without
having to call `nlp_var_col_map` and then scatter indices by hand.
"""
var_inds(model::JuMP.Model) = Dict(
    k => map(v -> JuMP.index(v).value, v)
    for (k, v) in object_dictionary(model)
    if v isa AbstractArray{VariableRef}
)

function _reformulation_helpers(model)
    inv_var_map = Dict(col => vi for (vi, col) in nlp_var_col_map(model))
    inv_con_map = Dict(row => ci for (ci, row) in nlp_con_row_map(model))

    get_reference(i, t) = if t == :var
        JuMP.VariableRef(model, inv_var_map[i])
    elseif t == :con
        JuMP.ConstraintRef(model, inv_con_map[i], JuMP.ScalarShape())
    else
        error("Unsupported complementarity type: $t")
    end

    get_cons_expr(con::JuMP.ConstraintRef) = begin
        obj = constraint_object(con)
        expr = obj.func - MOI.constant(obj.set)
        return expr
    end

    # Maps constraint row index -> slack variable, so repeated occurrences of
    # the same constraint reuse the existing slack (the original constraint is
    # deleted on first encounter and can't be referenced again).
    con_slack_cache = Dict{Int, JuMP.VariableRef}()

    get_or_create_con_slack!(row_idx) = get!(con_slack_cache, row_idx) do
        con = get_reference(row_idx, :con)
        slack = @variable(model, lower_bound=0)
        @constraint(model, slack == get_cons_expr(con))
        delete(model, con)
        slack
    end

    return get_reference, get_or_create_con_slack!, con_slack_cache
end

"""
    reformulate_sos1(model, ind_cc1, ind_cc2, types)::JuMP.Model

Reformulate complementarity constraints as SOS1 sets in-place.

Each triplet `(i1, i2, (t1, t2))` drawn from `zip(ind_cc1, ind_cc2, types)`
encodes one complementarity condition `x1 ⟂ x2`, where:

- `t1`, `t2` ∈ `{:var, :con}` indicate whether the index refers to a variable
  column or a constraint row (as returned by `nlp_var_col_map` /
  `nlp_con_row_map`).
- `i1`, `i2` are the corresponding 1-based column/row indices.

Behaviour by case:

| `(t1, t2)`    | Action |
|---------------|--------|
| `(:var, :var)` | Add `[x1, x2] ∈ SOS1` directly. |
| `(:var, :con)` | Introduce a non-negative slack `s = expr(con)`, delete the original constraint, add `[x1, s] ∈ SOS1`. |
| `(:con, :var)` | Symmetric to `(:var, :con)`. |
| `(:con, :con)` | Introduce two non-negative slacks, delete both original constraints, add `[s1, s2] ∈ SOS1`. |

The constraint expression used for `:con` entries is `func - constant(set)`,
i.e. the left-hand side when the constraint is written as `expr ∈ set`.
"""
function reformulate_sos1(model::JuMP.Model, ind_cc1, ind_cc2, types)::JuMP.Model
    new_model = copy(model) # Work on a copy to avoid modifying the original model's indices
    get_reference, get_or_create_con_slack!, _ = _reformulation_helpers(new_model)

    add_cc_pair!(i1, i2, t1, t2) = begin
        if (t1, t2) == (:var, :var)
            ref1 = get_reference(i1, t1)
            ref2 = get_reference(i2, t2)
            @constraint(new_model, [ref1, ref2] in JuMP.SOS1())
        elseif (t1, t2) == (:var, :con)
            ref1 = get_reference(i1, :var)
            slack = get_or_create_con_slack!(i2)
            @constraint(new_model, [ref1, slack] in JuMP.SOS1())
        elseif (t1, t2) == (:con, :var)
            add_cc_pair!(i2, i1, :var, :con)
        elseif (t1, t2) == (:con, :con)
            slack1 = get_or_create_con_slack!(i1)
            slack2 = get_or_create_con_slack!(i2)
            @constraint(new_model, [slack1, slack2] in JuMP.SOS1())
        else
            error("Unsupported complementarity type: ($t1, $t2)")
        end
    end

    foreach(((i1, i2, (t1, t2)),) -> add_cc_pair!(i1, i2, t1, t2), zip(ind_cc1, ind_cc2, types))

    return new_model
end

function reformulate_lie(model::JuMP.Model, ind_cc1, ind_cc2, types; κ, form=:eq)::JuMP.Model
    new_model = copy(model)
    get_reference, _, _ = _reformulation_helpers(new_model)

    ncc = length(ind_cc1)
    # retraction(x) = 1/2 * (x + sqrt(x^2 + 4))
    # inv_retraction(y) = y - 1/2 * (y - sqrt(y^2 + 4))

    retraction(x) = exp(x)
    inv_retraction(y) = log(y)
    
    con_expr(ref)  = (o = constraint_object(ref); o.func - MOI.constant(o.set))

    @variable(new_model, s1[1:ncc])
    if form == :ineq
        @variable(new_model, s2[1:ncc])
        @constraint(new_model, s1 + s2 .<= inv_retraction(κ))
    end

    for (j, (i1, i2, (t1, t2))) in enumerate(zip(ind_cc1, ind_cc2, types))
        cc1 = get_reference(i1, t1)
        cc2 = get_reference(i2, t2)

        expr1 = t1 == :var ? cc1 : con_expr(cc1)
        expr2 = t2 == :var ? cc2 : con_expr(cc2)

        t1 == :var ? delete_lower_bound(cc1) : delete(new_model, cc1)
        t2 == :var ? delete_lower_bound(cc2) : delete(new_model, cc2)

        if form == :ineq
            @constraint(new_model, expr1 == retraction(s1[j]))
            @constraint(new_model, expr2 == retraction(s2[j]))
        else
            @error "Only :ineq form is currently implemented"
        end
    end

    return new_model
end
