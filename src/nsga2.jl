"""
Non-dominated Sorting Genetic Algorithm (NSGA-II) for Multi-objective Optimization

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `crossoverRate`: The fraction of the population at the next generation, that is created by the crossover function
- `mutationRate`: Probability of chromosome to be mutated
- `selection`: [Selection](@ref) function (default: `tournament`)
- `crossover`: [Crossover](@ref) function (default: `SBX`)
- `mutation`: [Mutation](@ref) function (default: `PLM`)
- `metrics` is a collection of convergence metrics.
"""
struct NSGA2{T1, T2, T3} <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    selection::T1
    crossover::T2
    mutation::T3
    metrics::ConvergenceMetrics

    NSGA2(;
        populationSize::Int = 50, crossoverRate::Float64 = 0.9, mutationRate::Float64 = 0.1,
        selection::T1 = tournament(2, select = twowaycomp),
        crossover::T2 = SBX(),
        mutation::T3 = PLM(),
        metrics = ConvergenceMetric[GD(), GD(true)]
    ) where {T1, T2, T3} =
        new{T1, T2, T3}(
        populationSize, crossoverRate, mutationRate, selection,
        crossover, mutation, metrics
    )
end
population_size(method::NSGA2) = method.populationSize
default_options(method::NSGA2) = (iterations = 1000,)
summary(m::NSGA2) = "NSGA-II[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate)]"
show(io::IO, m::NSGA2) = print(io, summary(m))

mutable struct NSGAState{T, IT} <: AbstractOptimizerState
    N::Int                      # population size
    fitness::AbstractMatrix{T}  # fitness of the fittest individuals
    fitpop::AbstractMatrix{T}   # fitness of the whole population (including offspring)
    fittest::AbstractVector{IT} # fittest individuals
    offspring::AbstractArray    # offspring cache
    population::AbstractArray   # combined population (parents + offspring)
    ranks::Vector{Int}          # individual ranks
    crowding::Vector{T}         # individual crowding distance
end
value(s::NSGAState) = s.fitness
minimizer(s::NSGAState) = s.fittest

"""Initialization of NSGA2 algorithm state"""
function initial_state(method::NSGA2, options, objfun, parents)

    v = value(objfun) # objective function value
    T = eltype(v)     # objective function value type
    d = length(v)     # objective function value dimension
    N = length(first(parents)) # parents dimension
    IT = eltype(parents)       # individual type
    offspring = similar(parents) # offspring cache

    # construct fitness array that covers total population,
    # i.e. parents + offspring
    fitpop = fill(typemax(T), d, method.populationSize * 2)

    # Evaluate parents fitness
    value!(objfun, fitpop, parents)

    # setup initial state
    allpop = StackView(parents, offspring)
    ranks = vcat(fill(1, method.populationSize), fill(2, method.populationSize))
    crowding = vcat(fill(zero(T), method.populationSize), fill(typemax(T), method.populationSize))
    return NSGAState(N, zeros(T, d, 0), fitpop, IT[], offspring, allpop, ranks, crowding)
end

function update_state!(objfun, constraints, state, parents::AbstractVector{IT}, method::NSGA2, options, itr) where {IT}
    populationSize = method.populationSize
    rng = options.rng

    # select offspring
    specFit = StackView(state.ranks, state.crowding, dims = 1)
    selected = method.selection(view(specFit, :, 1:populationSize), populationSize; rng = rng)

    # perform mating
    recombine!(state.offspring, parents, selected, method, rng = rng)

    # perform mutation
    mutate!(state.offspring, method, constraints, rng = rng)

    # calculate fitness of the offspring
    offfit = @view state.fitpop[:, (populationSize + 1):end]
    evaluate!(objfun, offfit, state.offspring, constraints)

    # calculate ranks & crowding for population
    F = nondominatedsort!(state.ranks, state.fitpop)
    crowding_distance!(state.crowding, state.fitpop, F)

    # select best individuals
    fitidx = Int[]
    for f in F
        if length(fitidx) + length(f) > populationSize
            idxs = sortperm(view(state.crowding, f), rev = true)
            append!(fitidx, f[idxs[1:(populationSize - length(fitidx))]])
            break
        else
            append!(fitidx, f)
        end
    end
    # designate the first Pareto front individuals as the fittest
    fidx = length(F[1]) > populationSize ? fitidx : F[1]
    state.fittest = state.population[fidx]
    # and keep their fitness
    state.fitness = state.fitpop[:, fidx]

    # construct new parent population, keeping objective values / ranks /
    # crowding distances aligned with the survivor genomes (temporary copies
    # so the in-place assignments cannot alias overlapping views)
    parents .= state.population[fitidx]
    fitkeep = state.fitpop[:, fitidx]
    rankkeep = state.ranks[fitidx]
    crowdkeep = state.crowding[fitidx]
    state.fitpop[:, 1:populationSize] .= fitkeep
    state.ranks[1:populationSize] .= rankkeep
    state.crowding[1:populationSize] .= crowdkeep

    return false
end
