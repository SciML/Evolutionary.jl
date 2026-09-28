"""
Implementation of Differential Evolution: DE/**selection**/n/**recombination**

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `F`: the differentiation (mutation) scale factor (default: 0.9). It's usually defined in range ``F \\in (0, 1+]``
- `n`: the number of differences used in the perturbation (default: 1)
- `selection`: the selection strategy function (default: [`random`](@ref))
- `recombination`: the recombination functions (default: [`BINX(0.5)`](@ref))
- `K`: the recombination scale factor (default: 0.5*(F+1))
- `metrics` is a collection of convergence metrics.
"""
@kwdef struct DE{T1, T2} <: AbstractOptimizer
    populationSize::Integer = 50
    F::Real = 0.9
    n::Integer = 1
    K::Real = 0.5 * (F + 1)
    selection::T1 = random
    recombination::T2 = BINX(0.5)
    metrics::ConvergenceMetrics = ConvergenceMetric[AbsDiff(1.0e-10)]
end
population_size(method::DE) = method.populationSize
default_options(method::DE) = (iterations = 1000,)
summary(m::DE) = "DE/$(m.selection)/$(m.n)/$(m.recombination)"
show(io::IO, m::DE) = print(io, summary(m))

mutable struct DEState{T, IT} <: AbstractOptimizerState
    N::Int
    fitness::Vector{T}
    offitness::Vector{T}
    fittest::IT
end
value(s::DEState) = minimum(s.fitness)
minimizer(s::DEState) = s.fittest

"""Initialization of DE algorithm state"""
function initial_state(method::DE, options, objfun, population)
    T = typeof(value(objfun))
    individual = first(population)
    N = length(individual)
    fitness = fill(maxintfloat(T), method.populationSize)
    offitness = fill(maxintfloat(T), method.populationSize)

    # setup initial state
    return DEState(N, fitness, offitness, copy(individual))
end

function update_state!(objfun, constraints, state, population::AbstractVector{IT}, method::DE, options, itr) where {IT}

    # setup
    Np = method.populationSize
    n = method.n
    F = method.F
    rng = options.rng

    offspring = Array{IT}(undef, Np)

    # select base vectors
    bases = method.selection(state.fitness, Np, rng = rng)

    # select target vectors
    for (i, b) in enumerate(bases)
        base = population[b]
        offspring[i] = copy(base)

        # mutation
        targets = randexcl(rng, 1:Np, [i], 2 * n)
        offspring[i] = differentiation(offspring[i], @view population[targets]; F = F)

        # recombination with the target vector
        offspring[i], _ = method.recombination(offspring[i], population[i], rng = rng)

        # apply constraints
        apply!(constraints, offspring[i])
    end

    # Evaluate new offspring
    evaluate!(objfun, state.offitness, offspring, constraints)

    # Create new generation
    fitidx = 0
    minfit = minimum(state.fitness)
    for i in 1:Np
        v = state.offitness[i]
        if (v <= state.fitness[i])
            population[i] = offspring[i]
            state.fitness[i] = v
            if v < minfit
                minfit = v
                fitidx = i
            end
        end
    end

    # set best individual
    if fitidx > 0
        state.fittest = population[fitidx]
    end

    return false
end
