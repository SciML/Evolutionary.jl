#######################
# Convergence Metrics #
#######################

# Utilities

abschange(curr, prev) = Float64(abs(curr - prev))
relchange(curr, prev) = Float64(abs(curr - prev)/abs(curr))

maxdiff(x::AbstractArray, y::AbstractArray) = mapreduce((a, b) -> abs(a - b), max, x, y)
abschange(curr::T, prev) where {T<:AbstractArray} = maxdiff(curr, prev)
relchange(curr::T, prev) where {T<:AbstractArray} = maxdiff(curr, prev)/maximum(abs, curr)

# Single value

"""
Absolute difference convergence for single objective optimization.

This convergence metric allows to estimate an absolute difference between consecutive
states of the optimization algorithm, and triggers convergence when,

- `|f(x) - f(x')| < ε`

where `ε` is a tolerance value, `x` and `x'` are previous and current minimizers
found by the optimization algorithm.
"""
mutable struct AbsDiff{T} <: ConvergenceMetric
    tol::T
    Δ::Float64
    value::T
end
AbsDiff(tol::T) where {T<:AbstractFloat} = AbsDiff(tol, Inf, zero(T))
AbsDiff() = AbsDiff(1e-12)
description(m::AbsDiff) = "|f(x) - f(x')|"
function assess!(m::AbsDiff, state::AbstractOptimizerState)
    val = value(state)
    m.Δ = abschange(val, m.value)
    m.value = val
    converged(m)
end

"""
Relative difference convergence metric for single objective optimization.

This convergence metric allows to estimate a relative difference between consecutive
states of the optimization algorithm, and triggers convergence when,

- `|f(x) - f(x')|/|f(x')| < ε`

where `ε` is a tolerance value, `x` and `x'` are previous and current minimizers
found by the optimization algorithm.
"""
mutable struct RelDiff{T} <: ConvergenceMetric
    tol::T
    Δ::Float64
    value::T
end
RelDiff(tol::T) where {T<:AbstractFloat} = RelDiff(tol, Inf, zero(T))
RelDiff() = RelDiff(1e-12)
description(m::RelDiff) = "|f(x) - f(x')|/|f(x')|"
function assess!(m::RelDiff, state::AbstractOptimizerState)
    val = value(state)
    m.Δ = relchange(val, m.value)
    m.value = val
    converged(m)
end


"""
    gd(A,R)

Calculate a generational distance between set `A` and the reference set `R`.
This metric measures the convergence, i.e. closeness of the non-dominated solutions
to the Pareto front, of a population.

*Note:* Parameters are column-major matrices.
"""
function gd(A::AbstractMatrix, R::AbstractMatrix)
    da, na = size(A)
    dr, nr = size(R)
    (na == 0 || nr == 0) && return Inf
    sum = 0
    for a in eachcol(A)
        sum += minimum(norm(a-r) for r in eachcol(R))
    end
    sum/na
end

"""
    igd(S,R)

Calculate an inverted generational distance, [`gd`](@ref), between set `S` and the reference set `R`.
Parameters are column-major matrices.
"""
igd(S,R) = gd(R,S)


"""
(Inverse) generational distance convergence metric for multi-objective optimization.

This convergence metric allows to estimate a (inverse) generational distance between
consecutive states of the multi-objective optimization algorithm, and triggers convergence when,

- ``GD(S,R) = \\frac{1}{|S|}\\left(\\sum_{s \\in S} \\min_{r \\in R}||f(s) - f(r)||^2\\right)^{1/2} < \\varepsilon``

where `ε` is a tolerance value, `S` and `R` are discrete Pareto front approximations
for previous and current state of the optimization algorithm.

The inverse generational distance calculated as `IGD(S,R) = GR(R,S)`.
To use, `IGD` metric pass `true` value to the metric constructor argument, e.g.

- `GD()` create a generational distance metric
- `GD(true)` create an inverse generational metric
"""
mutable struct GD{T} <: ConvergenceMetric
    tol::T
    Δ::Float64
    value::Matrix{T}
    inverted::Bool
end
GD(tol::T, inv=false) where {T<:AbstractFloat} = GD(tol, Inf, zeros(T,0,0), inv)
GD(inv=false) = GD(1e-5, inv)
function description(m::GD)
    prefix = m.inverted ? "I" : ""
    "|$(prefix)GD(P) - $(prefix)GD(P')|"
end
function assess!(m::GD, state::AbstractOptimizerState)
    val = value(state)
    m.Δ = m.inverted ? igd(val, m.value) : gd(val, m.value)
    m.value = val
    converged(m)
end

"""
"Never converges" metric for single objective optimization.

This allows one to force the optimization algorithm to run for a given number of iterations.
"""
mutable struct Never <: ConvergenceMetric
end
description(m::Never) = "never converges"
assess!(m::Never, state::AbstractOptimizerState) = false


"""
    spread(S,R)

Returns a diversity metric of a population of set `S` to the reference set `R`.
"""
function spread(S::AbstractMatrix, R::AbstractMatrix)
    n = size(S,2)
    m = size(R,2)
    Δₖ = [ minimum(norm(view(S,:,i)-view(R,:,j)) for i in 1:n if view(S,:,i) != view(R,:,j)) for j in 1:m ]
    Δ = mean(Δₖ)
    sum(abs.(Δₖ.-Δ))/m*Δ
end

function spread(S::AbstractMatrix)
    n = size(S,2)
    n == 1 && return NaN
    Δₖ = [ minimum(norm(view(S,:,i)-view(S,:,j)) for j in 1:n if view(S,:,i) != view(S,:,j)) for i in 1:n ]
    Δ = mean(Δₖ)
    sum(abs.(Δₖ.-Δ))/n*Δ
end

##########################
# Convergence Assessment #
##########################

assess_convergence(state::AbstractOptimizerState, method) =
    any(assess!(cm, state) for cm in metrics(method))

