"""
Implementation of Genetic Algorithm

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `crossoverRate`: The fraction of the population at the next generation, not including elite children, that is created by the crossover function.
- `mutationRate`: Probability of chromosome to be mutated
- `ɛ`/`epsilon`: Positive integer specifies how many individuals in the current generation are guaranteed to survive to the next generation.
Floating number specifies fraction of population.
- `selection`: [Selection](@ref) function (default: [`tournament`](@ref))
- `crossover`: [Crossover](@ref) function (default: [`genop`](@ref))
- `mutation`: [Mutation](@ref) function (default: [`genop`](@ref))
- `after_op`: a function that is executed on each individual after mutation operations (default: `identity`)
- `metrics` is a collection of convergence metrics.
"""
struct GA{T1, T2, T3, T4} <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::T1
    crossover::T2
    mutation::T3
    after_op::T4
    metrics::ConvergenceMetrics

    GA(;
        populationSize::Int = 50, crossoverRate::Float64 = 0.8, mutationRate::Float64 = 0.1,
        ɛ::Real = 0, epsilon::Real = ɛ,
        selection::T1 = tournament(2),
        crossover::T2 = genop,
        mutation::T3 = genop,
        after_op::T4 = identity,
        metrics = ConvergenceMetric[AbsDiff(1.0e-12)]
    ) where {T1, T2, T3, T4} =
        new{T1, T2, T3, T4}(populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation, after_op, metrics)
end
population_size(method::GA) = method.populationSize
"""
    default_options(method)

Return the default optimization options for an evolutionary method.
"""
default_options(method::GA) = (iterations = 1000,)
summary(m::GA) = "GA[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate),ɛ=$(m.ɛ)]"
show(io::IO, m::GA) = print(io, summary(m))

mutable struct GAState{T, IT} <: AbstractOptimizerState
    N::Int
    eliteSize::Int
    fitness::T
    fitpop::Vector{T}
    fittest::IT
end
value(s::GAState) = s.fitness
minimizer(s::GAState) = s.fittest

"""Initialization of GA algorithm state"""
function initial_state(method::GA, options, objfun, population)
    T = typeof(value(objfun))
    N = first(population) isa Expr ? nodes(first(population)) : length(first(population))

    # evaluate elite size and extend population
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)
    if eliteSize > 0
        for i in 1:eliteSize
            push!(population, copy(first(population)))
        end
    end

    # create fitness values
    fitness = zeros(T, method.populationSize + eliteSize)

    # Evaluate population fitness
    fitness = map(i -> value(objfun, i), population)
    minfit, fitidx = findmin(fitness)

    # setup initial state
    return GAState(N, eliteSize, minfit, fitness, copy(population[fitidx]))
end

function update_state!(objfun, constraints, state, parents::AbstractVector{IT}, method::GA, options, itr) where {IT}
    populationSize = method.populationSize
    rng = options.rng

    # create an offspring popultation
    offspring = similar(parents)

    # select offspring
    selected = method.selection(state.fitpop, populationSize, rng = rng)

    # perform mating
    recombine!(offspring, parents, selected, method, rng = rng)

    # perform mutation
    mutate!(view(offspring, 1:populationSize), method, constraints, rng = rng)

    # Elitism (copy elite individuals from selection to the offspring)
    selfit = view(state.fitpop, selected)
    fitidxs = sortperm(selfit)
    for i in 1:state.eliteSize
        subs = populationSize + i
        offspring[subs] = parents[selected[fitidxs[i]]]
    end

    # calculate fitness of the population
    evaluate!(objfun, state.fitpop, offspring, constraints)

    # apply auxiliary function after mutation operations
    method.after_op !== identity && broadcast!(method.after_op, offspring, offspring)

    # select the best individual
    minfit, fitidx = findmin(state.fitpop)
    state.fittest = offspring[fitidx]
    state.fitness = state.fitpop[fitidx]

    # replace population
    parents .= offspring

    return false
end

function recombine!(
        offspring, parents, selected, method;
        rng::AbstractRNG = default_rng()
    )
    n = length(selected)
    mates = ((i, i == n ? i - 1 : i + 1) for i in 1:2:n)
    for (i, j) in mates
        p1, p2 = parents[selected[i]], parents[selected[j]]
        if rand(rng) < method.crossoverRate
            offspring[i], offspring[j] = method.crossover(p1, p2, rng = rng)
        else
            offspring[i], offspring[j] = p1, p2
        end
    end

    return
end

function mutate!(
        population, method, constraints;
        rng::AbstractRNG = default_rng()
    )
    n = length(population)
    for i in 1:n
        if rand(rng) < method.mutationRate
            method.mutation(population[i], rng = rng)
        end
        apply!(constraints, population[i])
    end
    return
end

function evaluate!(objfun, fitness, population, constraints)
    # calculate fitness of the population
    value!(objfun, fitness, population)
    # apply penalty to fitness
    return penalty!(fitness, constraints, population)
end
