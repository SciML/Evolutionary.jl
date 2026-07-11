# Individual
const Individual = Union{AbstractArray, Function, Nothing}


# Optimizer
"""
Abstract evolutionary optimizer algorithm
"""
abstract type AbstractOptimizer end

"""
    default_options(method)

Return the default [`Options`](@ref) keyword values for an evolutionary
optimization method.

Algorithm implementations extend this function with a method for their
optimizer type. The returned named tuple is passed to `Options` when the user
does not provide explicit options.
"""
function default_options end

function print_header(method::AbstractOptimizer)
    return println("Iter     Function value")
end
population_size(method::AbstractOptimizer) = error("`population_size` is not implemented for $(summary(method)).")
metrics(method::AbstractOptimizer) = method.metrics

# Optimizer State
"""
Abstract type for defining an optimizer state

Every algorithm have to implement a state type derived from this abstract type.
"""
abstract type AbstractOptimizerState end

"""
    value(state)

Returns a minimum value of the current `state`.
"""
value(state::AbstractOptimizerState) = error("`value` is not implemented for $(state).")

"""
    minimizer(state)

Returns a minimizer object in the current `state`.
"""
minimizer(state::AbstractOptimizerState) = error("`minimizer` is not implemented for $(state).")


"""
    terminate(state)

Returns `true` if the `state` requires early termination.
"""
terminate(state::AbstractOptimizerState) = false


# Convergence
"""
Interface for the convergence metrics
"""
abstract type ConvergenceMetric end

"""
    description(metric)

Return a string with a description of the `metric`.
"""
description(m::ConvergenceMetric) = error("`description` is not implemented for $(typeof(m)).")

"""
    converged(metric)

Return `true` if the convergence is archived for the `metric`.
"""
converged(m::ConvergenceMetric) = diff(m) <= tolerance(m)

"""
    diff(metric)

Return the value difference for the `metric`.
"""
diff(m::ConvergenceMetric) = m.Δ #error("`diff` is not implemented for $(typeof(m)).")

"""
    tolerance(metric)

Return a tolerance value for the `metric`.
"""
tolerance(m::ConvergenceMetric) = m.tol #error("`tolerance` is not implemented for $(typeof(m)).")

"""
    assess!(metric, state)

Asses the convergence of an algorithm using the `metric`at the `state`.
"""
assess!(m::ConvergenceMetric, s::AbstractOptimizerState) = error("`assess!` is not implemented for $(typeof(m)).")


const ConvergenceMetrics = Vector{ConvergenceMetric}

# Options
"""
There are following options available:
- `abstol::Float64`: the absolute tolerance used in the convergence test
- `reltol::Float64`: the relative tolerance used in the convergence test
- `successive_f_tol::Integer`: the additional number of the iterations of the optimization algorithm after the convergence test is satisfied (*default: 10*)
- `iterations::Integer`: the total number of the iterations of the optimization algorithm (*default: 1000*)
- `show_trace::Bool`: enable the trace information display during the optimization (*default: false*).
- `store_trace::Bool`: enable the trace information capturing during the optimization (*default: false*). The trace can be accessed by using [`trace`](@ref) function after optimization is finished.
- `show_every::Integer`: show every `n`s successive trace message (*default: 1*)
- `time_limit::Float64`: the time limit for the optimization run in seconds. If the value set to `NaN` then the limit is not set. (*default: NaN*)
- `callback`: the callback function that is called after each iteration of the optimization algorithm. The function accepts as parameter a trace dictionary, and **must** return a `Bool` value which if `true` terminates the optimization. (*default: nothing*)
- `parallelization::Symbol`: allows parallelization of the population fitness evaluation if set to `:thread` using multiple threads (*default: `:serial`*)
- `rng::AbstractRNG`: a random number generator object that is used to control generation of random data during the evolutionary optimization (*default: `Random.default_rng()`*)
"""
@kwdef struct Options{TCallback <: Union{Nothing, Function}, TRNG <: AbstractRNG}
    abstol::Float64 = Inf
    reltol::Float64 = Inf
    successive_f_tol::Int = 10
    iterations::Int = 1000
    store_trace::Bool = false
    show_trace::Bool = false
    show_every::Int = 1
    callback::TCallback = nothing
    time_limit::Float64 = NaN
    parallelization::Symbol = :serial
    rng::TRNG = default_rng()
end
function show(io::IO, o::Options)
    for k in fieldnames(typeof(o))
        v = getfield(o, k)
        if v === nothing
            print(io, lpad("$(k)", 24) * " = nothing\n")
        else
            print(io, lpad("$(k)", 24) * " = $v\n")
        end
    end
    return
end


# Optimization Trace

struct OptimizationTraceRecord{T, O <: AbstractOptimizer}
    iteration::Int
    value::T
    metadata::Dict{String, Any}
end
value(tr::OptimizationTraceRecord) = tr.value

function show(io::IO, t::OptimizationTraceRecord)
    print(io, lpad("$(t.iteration)", 6))
    print(io, "   ")
    print(io, lpad("$(t.value)", 14))
    for (key, value) in t.metadata
        print(io, "\n * $key: $value")
    end
    return
end

const OptimizationTrace{T, O} = Vector{OptimizationTraceRecord{T, O}}

function show(io::IO, tr::OptimizationTrace)
    print(io, "Iter     Function value\n")
    print(io, "------   --------------\n")
    for rec in tr
        show(io, rec)
        print(io, "\n")
    end
    return
end


# Evolution strategies

"""Abstract evolution strategy

All evolution strategies must be derived from this type.
"""
abstract type AbstractStrategy end

"""
    strategy(state)

Return the evolution strategy associated with an optimizer state.
"""
function strategy end

"""Empty evolution strategy"""
struct NoStrategy <: AbstractStrategy end
copy(s::NoStrategy) = NoStrategy()

"""
Isotropic evolution strategy

This strategy has one mutation parameter for all object parameter components.
"""
mutable struct IsotropicStrategy{T <: Real} <: AbstractStrategy
    σ::T
    τ₀::T
    τ::T
end

"""
    IsotropicStrategy(N)

Returns an isotropic strategy object, which has an one mutation parameter for all object parameter components, with ``\\sigma = 1.0``, ``\\tau_0 = \\sqrt{2N}^{-1}``, ``\\tau = \\sqrt{2\\sqrt{N}}^{-1}``
"""
IsotropicStrategy(N::Integer) = IsotropicStrategy{Float64}(1.0, 1.0 / sqrt(2N), 1.0 / sqrt(2 * sqrt(N)))
copy(s::IsotropicStrategy) = IsotropicStrategy{typeof(s.σ)}(s.σ, s.τ₀, s.τ)

"""
Anisotropic evolution strategy

This strategy has a mutation parameter for each object parameter component.
"""
mutable struct AnisotropicStrategy{T} <: AbstractStrategy
    σ::Vector{T}
    τ₀::T
    τ::T
end

"""
    AnisotropicStrategy(N)

Returns an anisotropic strategy object, which has an one mutation parameter for each object parameter component, with ``\\sigma = [1, \\ldots, 1]^N``, ``\\tau_0 = \\sqrt{2N}^{-1}``, ``\\tau = \\sqrt{2\\sqrt{N}}^{-1}``
"""
AnisotropicStrategy(N::Integer) = AnisotropicStrategy(ones(N), 1 / sqrt(2N), 1 / sqrt(2 * sqrt(N)))
copy(s::AnisotropicStrategy) = AnisotropicStrategy{typeof(s.τ)}(copy(s.σ), s.τ₀, s.τ)
