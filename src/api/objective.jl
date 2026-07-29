"""
Wrapper around an objective function (compatible with NLSolversBase).
"""
mutable struct EvolutionaryObjective{TC, TF, TX, TP} <: AbstractObjective
    f::TC
    F::TF
    x_f::TX
    f_calls::Int
end

"""
    EvolutionaryObjective(f, x[, F])

Constructor for an objective function object around the function `f` with initial parameter `x`, and objective value `F`.
"""
function EvolutionaryObjective(
        f::TC, x::AbstractArray,
        F::Union{Real, AbstractArray{<:Real}} = zero(f(x));
        eval::Symbol = :serial
    ) where {TC}
    defval = default_values(x)
    # convert function into the in-place one
    TF = typeof(F)
    fn, TN = if funargnum(f) == 2 && F isa AbstractArray
        ff = (Fv, xv) -> (Fv .= f(xv))
        ff, typeof(ff)
    else
        f, TC
    end
    return EvolutionaryObjective{TN, TF, typeof(x), Val{eval}}(fn, F, defval, 0)
end

"""
    EvolutionaryObjective(f, x::Expr[, F])

Constructor for an objective object for a Julia evaluatable expression.
"""
function EvolutionaryObjective(
        f::TC, x::Expr, F::TF = zero(f(x));
        eval::Symbol = :serial
    ) where {TC, TF <: Real}
    return EvolutionaryObjective{TC, TF, typeof(x), Val{eval}}(f, F, :(), 0)
end

"""
    f_calls(obj::EvolutionaryObjective)

Return the number of objective evaluations performed by `obj`.

# Arguments
- `obj`: Objective wrapper whose evaluation count is queried.

# Examples
```julia
julia> obj = EvolutionaryObjective(sum, zeros(2));

julia> value!(obj, [1.0, 2.0]); f_calls(obj)
1
```
"""
f_calls(obj::EvolutionaryObjective) = obj.f_calls

"""
    value(obj::EvolutionaryObjective)
    value(obj::EvolutionaryObjective, x)

Return the cached objective value of `obj`, or evaluate the objective at `x`.
Evaluation increments [`f_calls`](@ref).

# Arguments
- `obj`: Objective wrapper to inspect or evaluate.
- `x`: Candidate point supplied to the wrapped objective.

# Examples
```julia
julia> obj = EvolutionaryObjective(sum, zeros(2));

julia> value(obj, [1.0, 2.0])
3.0
```
"""
value(obj::EvolutionaryObjective) = obj.F

"""
    ismultiobjective(objfun)

Return `true` if the function is multi-objective objective.
"""
ismultiobjective(obj) = obj.F isa AbstractArray

function value(obj::EvolutionaryObjective{TC, TF, TX, TP}, x::TX) where {TC, TF, TX, TP}
    obj.f_calls += 1
    return obj.f(x)::TF
end

"""
    value!(obj::EvolutionaryObjective, x)
    value!(obj::EvolutionaryObjective, values, population)

Evaluate an `EvolutionaryObjective`, store the result in its cache, and return the
stored value. With `values` and `population`, evaluate each population member into
the supplied output array.

# Arguments
- `obj`: Objective wrapper to evaluate.
- `x`: Candidate point to evaluate.
- `values`: Preallocated output array for population evaluation.
- `population`: Candidate points to evaluate.

# Examples
```julia
julia> obj = EvolutionaryObjective(sum, zeros(2));

julia> value!(obj, [1.0, 2.0])
3.0
```
"""
function value!(obj::EvolutionaryObjective{TC, TF, TX, TP}, x::TX) where {TC, TF, TX, TP}
    obj.F = value(obj, x)
    return obj.F
end

"""
    value!!(obj::EvolutionaryObjective, x)
    value!!(obj::EvolutionaryObjective, values, x)

Evaluate an `EvolutionaryObjective` after copying `x` into its cached input, then
return the objective value. Use this variant when later optimizer steps need the
evaluated candidate retained by the objective wrapper.

# Arguments
- `obj`: Objective wrapper to evaluate and update.
- `x`: Candidate point to cache and evaluate.
- `values`: Workspace used by an in-place objective evaluation.

# Examples
```julia
julia> obj = EvolutionaryObjective(sum, zeros(2));

julia> value!!(obj, [1.0, 2.0])
3.0
```
"""
function value!!(obj::EvolutionaryObjective{TC, TF, TX, TP}, x::TX) where {TC, TF, TX, TP}
    copyto!(obj.x_f, x)
    return value!(obj, x)
end

function value(obj::EvolutionaryObjective{TC, TF, TX, TP}, F, x::TX) where {TC, TF, TX, TP}
    obj.f_calls += 1
    return obj.f(F, x)
end
value(obj::EvolutionaryObjective{TC, TF, TX, TP}, x::TX) where {TC, TF <: AbstractArray, TX, TP} = value(obj, copy(obj.F), x)

function value!(obj::EvolutionaryObjective{TC, TF, TX, TP}, F, x::TX) where {TC, TF, TX, TP}
    return obj.F = value(obj, F, x)
end

function value!!(obj::EvolutionaryObjective{TC, TF, TX, TP}, F, x::TX) where {TC, TF, TX, TP}
    copyto!(obj.x_f, x)
    return value!(obj, F, x)
end

function value!(
        obj::EvolutionaryObjective{TC, TF, TX, Val{:serial}},
        F::AbstractVector, xs::AbstractVector{TX}
    ) where {TC, TF <: Real, TX}
    broadcast!(x -> value(obj, x), F, xs)
    return F
end

function value!(
        obj::EvolutionaryObjective{TC, TF, TX, Val{:thread}},
        F::AbstractVector, xs::AbstractVector{TX}
    ) where {TC, TF <: Real, TX}
    n = length(xs)
    Threads.@threads for i in 1:n
        F[i] = value(obj, xs[i])
    end
    return F
end

function value!(
        obj::EvolutionaryObjective{TC, TF, TX, Val{:serial}},
        F::AbstractMatrix, xs::AbstractVector{TX}
    ) where {TC, TF, TX}
    n = length(xs)
    for i in 1:n
        fv = view(F, :, i)
        value(obj, fv, xs[i])
    end
    return F
end

function value!(
        obj::EvolutionaryObjective{TC, TF, TX, Val{:thread}},
        F::AbstractMatrix, xs::AbstractVector{TX}
    ) where {TC, TF, TX}
    n = length(xs)
    @Threads.threads for i in 1:n
        fv = view(F, :, i)
        value(obj, fv, xs[i])
    end
    return F
end
