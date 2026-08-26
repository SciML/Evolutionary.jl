using Evolutionary
import NLSolversBase: value, value!
using Test
import Evolutionary: population_size, default_options, minimizer,
    initial_state, update_state!

# mock state
mutable struct TestOptimizerState <: Evolutionary.AbstractOptimizerState
    individual
    fitness
end
Evolutionary.value(state::TestOptimizerState) = state.fitness
Evolutionary.minimizer(state::TestOptimizerState) = state.individual


# mock optimizer
Base.@kwdef struct TestOptimizer <: Evolutionary.AbstractOptimizer
    metrics::Vector{ConvergenceMetric} = [Evolutionary.AbsDiff(1.0e-10)]
end
Evolutionary.population_size(method::TestOptimizer) = 5
Evolutionary.default_options(method::TestOptimizer) = Dict(:iterations => 10)
Evolutionary.initial_state(method, options, d, population) =
    TestOptimizerState(population[end], value(d, population[end]))
function Evolutionary.update_state!(d, constraints, state::TestOptimizerState, population::AbstractVector, method, opts, itr)
    i = rand(1:population_size(method))
    state.individual = population[i]
    state.fitness = value!(d, state.individual)
    return false
end
