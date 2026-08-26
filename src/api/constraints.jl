# Public interface for AbstractConstraints

"""
    constraint_values(c::AbstractConstraints, x)

Return the nonlinear constraint-function values for candidate `x` under `c`.

# Interface

Custom `AbstractConstraints` implementations with nonlinear constraints must extend
this function. Return `nothing` when no nonlinear constraint function is present.
Otherwise, return an indexable collection ordered consistently with the constraint
bounds returned by [`bounds`](@ref): equality values first, then inequality values.

# Arguments

- `c`: Constraint object being evaluated.
- `x`: Candidate optimization variable.

# Returns

`nothing` for constraints without nonlinear constraint functions, or the values used
by [`isfeasible`](@ref) and [`penalty`](@ref) to evaluate `c`.

# Examples

```julia
julia> constraint_values(NoConstraints(), [1.0, 2.0])
nothing
```
"""
constraint_values(c::AbstractConstraints, x) = nothing

"""
    penalty(constraints, x)

Calculate a penalty for the variable `x` given the set of `constraints`.
"""
penalty(c::AbstractConstraints, x) = 0

"""
    penalty!(fitness, constraints, population)

Set a penalty to the `fitness` given the set of `constraints` and `population`.
"""
penalty!(fitness, c::AbstractConstraints, population) = fitness

"""
    apply!(c::AbstractConstraints, x)

Appliy constrains `c` to a variable `x`, and return the variable.
"""
apply!(c::AbstractConstraints, x) = x

"""
    bounds(c::AbstractConstraints)

Return bounds for the constraint `c`.
"""
bounds(c::AbstractConstraints) = error("`bounds` is not implemented for $(c).")

# Auxiliary functions

constraint_dimension(bounds::ConstraintBounds) = max(
    isempty(bounds.eqx) ? 0 : maximum(bounds.eqx),
    isempty(bounds.ineqx) ? 0 : maximum(bounds.ineqx),
)

constraint_count(bounds::ConstraintBounds) = max(
    isempty(bounds.eqc) ? 0 : maximum(bounds.eqc),
    isempty(bounds.ineqc) ? 0 : maximum(bounds.ineqc),
)

"""
    isfeasible(bounds::ConstraintBounds, x) -> Bool

Return `true` if point `x` is feasible, given the `bounds` object with bounds `lx`, `ux`, `lc`, and `uc`. `x` is feasible if

    lx[i] <= x[i] <= ux[i]
    lc[i] <= c(x)[i] <= uc[i]

for all possible `i`.
"""
function isfeasible(bounds::ConstraintBounds, x::AbstractVector, c::Union{AbstractVector, Nothing} = nothing)
    isf = true
    for (i, j) in enumerate(bounds.eqx)
        isf &= x[j] == bounds.valx[i]
    end
    for (i, j) in enumerate(bounds.ineqx)
        isf &= bounds.σx[i] * (x[j] - bounds.bx[i]) >= 0
    end
    if c !== nothing
        for (i, j) in enumerate(bounds.eqc)
            isf &= c[j] == bounds.valc[i]
        end
        for (i, j) in enumerate(bounds.ineqc)
            isf &= bounds.σc[i] * (c[j] - bounds.bc[i]) >= 0
        end
    end
    return isf
end

"""
    isfeasible(c::AbstractConstraints, x) -> Bool

Return `true` if point `x` is feasible, given the constraints object `c`.
"""
isfeasible(c::AbstractConstraints, x) = isfeasible(bounds(c), x, constraint_values(c, x))

# Implementations

"""Type for an empty set of constraints"""
struct NoConstraints <: AbstractConstraints end
isfeasible(c::NoConstraints, x) = true
getproperty(c::NoConstraints, s::Symbol) =
    s == :bounds ? ConstraintBounds(Float64[], Float64[], Float64[], Float64[]) : nothing

"""
This type encodes box constraints for the optimization function parameters.

The constructor takes following arguments:

- `lower` is the vector of value lower bounds
- `upper` is the vector of value upper bounds

*Note: Sizes of the lower and upper bound vectors must be equal.*
"""
struct BoxConstraints{T} <: AbstractConstraints
    bounds::ConstraintBounds{T}
end
BoxConstraints(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T} =
    BoxConstraints(ConstraintBounds(lower, upper, [], []))
BoxConstraints(lower::T, upper::T, size) where {T} =
    BoxConstraints(fill(lower, size), fill(upper, size))
apply!(c::BoxConstraints, x) = clip!(c.bounds, x)
bounds(c::BoxConstraints) = c.bounds
function show(io::IO, c::BoxConstraints)
    print(io, "Box Constraints:")
    indent = "    "
    cb = bounds(c)
    if !isempty(cb.eqx)
        print(io, '\n', indent)
        join(io, ("x[$i]=$v" for (i, v) in zip(cb.eqx, cb.valx)), ", ")
    end
    return if !isempty(cb.ineqx)
        print(io, '\n', indent)
        join(io, ("x[$i]$(σ > 0 ? '≥' : '≤')$b" for (i, σ, b) in zip(cb.ineqx, cb.σx, cb.bx)), ", ")
    end
end

"""
This type encodes constraints as the following additional penalty for an objective function:

``p(x) = \\sum^n_{i=1} r_i max(0, g_i(x))^2``

where ``r_i`` is a penalty value for ``i``th constraint, and ``g_i(x)`` is an inequality constraint.
The equality constraints ``h_i(x) = 0`` are transformed to inequality constraints as follows:

``h(x) - \\epsilon  \\leq 0``

The constructor takes following arguments:

- `penalties`: a vector of penalty values per constraint (optional)
- `lx`: a vector of value lower bounds
- `ux`: a vector of value upper bounds
- `lc`: a vector of constrain function lower bounds
- `uc`: a vector of constrain function upper bounds
- `c`: a constraint function which returns a constrain values
"""
struct PenaltyConstraints{T, F} <: AbstractConstraints
    coef::Vector{T}
    constraints::F   # constraints(x) returns the value of the constraint-functions at x
    bounds::ConstraintBounds{T}
    PenaltyConstraints(penalty::Vector{T}, bounds::ConstraintBounds{T}, c::F) where {T, F} =
        new{T, F}(penalty, c, bounds)
end
PenaltyConstraints(penalty::T, bounds::ConstraintBounds{T}, cf = (x) -> nothing) where {T} =
    PenaltyConstraints(fill(penalty, constraint_count(bounds) + constraint_dimension(bounds)), bounds, cf)
PenaltyConstraints(
    penalty::AbstractVector{T}, lx::AbstractVector{T}, ux::AbstractVector{T},
    lc::AbstractVector{T}, uc::AbstractVector{T}, cf = (x) -> nothing
) where {T} =
    PenaltyConstraints(penalty, ConstraintBounds(lx, ux, lc, uc), cf)
PenaltyConstraints(
    penalty::T, lx::AbstractVector{T}, ux::AbstractVector{T},
    lc::AbstractVector{T} = T[], uc::AbstractVector{T} = T[],
    cf = (x) -> nothing
) where {T} =
    PenaltyConstraints(penalty, ConstraintBounds(lx, ux, lc, uc), cf)
PenaltyConstraints(
    lx::AbstractVector{T}, ux::AbstractVector{T},
    lc::AbstractVector{T} = T[], uc::AbstractVector{T} = T[],
    cf = (x) -> nothing
) where {T} =
    PenaltyConstraints(one(T), ConstraintBounds(lx, ux, lc, uc), cf)
constraint_values(c::PenaltyConstraints, x) = c.constraints(x)
bounds(c::PenaltyConstraints) = c.bounds
penalty(c::PenaltyConstraints, x) = penalty(c.bounds, c.coef, x, constraint_values(c, x))
function penalty!(fitness::AbstractVector{T}, c::PenaltyConstraints{T, F}, population) where {T, F}
    for (i, x) in enumerate(population)
        fitness[i] += penalty(c, x)
    end
    return fitness
end
function show(io::IO, c::PenaltyConstraints)
    println(io, "Penalty Constraints:")
    println(io, "  Penalties: $(c.coef)")
    return print(io, bounds(c))
end

"""
This type encodes constraints as the following additional penalty for an objective function:

``p(x) = f_{worst} + \\sum^n_{i=1} |g_i(x)|``

if ``x`` is not feasible, otherwise no penalty is applied.

The constructor takes following arguments:

- `lx`: a vector of value lower bounds
- `ux`: a vector of value upper bounds
- `lc`: a vector of constrain function lower bounds
- `uc`: a vector of constrain function upper bounds
- `c`: a constraint function which returns a constrain values
"""
struct WorstFitnessConstraints{T, F} <: AbstractConstraints
    bounds::ConstraintBounds{T}
    constraints::F   # constraints(x) returns the value of the constraint-functions at x
end
WorstFitnessConstraints(
    lx::AbstractVector{T}, ux::AbstractVector{T}, lc::AbstractVector{T},
    uc::AbstractVector{T}, cf = (x) -> nothing
) where {T} =
    WorstFitnessConstraints(ConstraintBounds(lx, ux, lc, uc), cf)
constraint_values(c::WorstFitnessConstraints, x) = c.constraints(x)
apply!(c::WorstFitnessConstraints, x) = clip!(c.bounds, x)
bounds(c::WorstFitnessConstraints) = c.bounds
function penalty!(fitness::AbstractVector{T}, c::WorstFitnessConstraints{T, F}, population) where {T, F}
    worst = maximum(fitness)
    for (i, x) in enumerate(population)
        cv = constraint_values(c, x)
        p = zeros(size(cv))
        if !isfeasible(c.bounds, x, cv)
            for (i, j) in enumerate(c.bounds.eqc)
                p[j] = cv[j] - c.bounds.valc[i] - eps()
            end
            for (i, j) in enumerate(c.bounds.ineqc)
                p[j] = c.bounds.σc[i] * (c.bounds.bc[i] - cv[j])
            end
            fitness[i] = worst + sum(abs, p)
        end
    end
    return fitness
end
function show(io::IO, c::WorstFitnessConstraints)
    print(io, "Worst Fitness ")
    return print(io, c.bounds)
end

"""
This type provides an additional type constraints on the variables required for mixed integer optimization problems.
"""
struct MixedTypePenaltyConstraints{C <: AbstractConstraints} <: AbstractConstraints
    penalty::C
    types::Vector{DataType}
end
constraint_values(c::MixedTypePenaltyConstraints, x) = constraint_values(c.penalty, x)
penalty!(fitness, c::MixedTypePenaltyConstraints, population) = penalty!(fitness, c.penalty, population)
function apply!(c::MixedTypePenaltyConstraints, x)
    y = apply!(c.penalty, x)
    for (i, t) in enumerate(c.types)
        if t <: Integer
            y[i] = round(t, y[i])
        end
    end
    return y
end

# Utilities

function clip!(bounds::ConstraintBounds{T}, x) where {T}
    for (i, j) in enumerate(bounds.eqx)
        x[j] = bounds.valx[i]
    end
    for (i, j) in enumerate(bounds.ineqx)
        if bounds.σx[i] * (x[j] - bounds.bx[i]) < 0
            x[j] = bounds.bx[i]
        end
    end
    return x
end

function penalty(
        bounds::ConstraintBounds{T}, coeff::AbstractVector{T},
        x, c::Union{AbstractVector{T}, Nothing} = nothing
    ) where {T}
    penalty = 0
    xc = constraint_dimension(bounds)
    for (i, j) in enumerate(bounds.eqx)
        penalty += coeff[j] * (x[j] - bounds.valx[i] - eps())^2
    end
    for (i, j) in enumerate(bounds.ineqx)
        penalty += coeff[j] * max(0, bounds.σx[i] * (bounds.bx[i] - x[j]))^2
    end
    if c !== nothing
        for (i, j) in enumerate(bounds.eqc)
            penalty += coeff[xc + j] * max(0, c[j] - bounds.valc[i] - eps())^2
        end
        for (i, j) in enumerate(bounds.ineqc)
            penalty += coeff[xc + j] * max(0, bounds.σc[i] * (bounds.bc[i] - c[j]))^2
        end
    end
    return penalty
end
