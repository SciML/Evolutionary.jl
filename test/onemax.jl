using Evolutionary
using Test
using Random
using StableRNGs

@testset "OneMax" begin

    rng = StableRNG(42)

    # Initial population
    N = 100
    initpop = (() -> BitArray(rand(rng, Bool, N)))

    function Evolutionary.trace!(record::Dict{String, Any}, objfun, state, population, method::GA, options)
        idx = sortperm(state.fitpop)
        record["fitpop"] = state.fitpop[idx[1:5]]
    end

    res = Evolutionary.optimize(
        x -> -sum(x),                 # Function to MINIMIZE
        initpop,
        GA(
            selection = tournament(3),
            mutation = flip,
            crossover = TPX,
            mutationRate = 0.05,
            crossoverRate = 0.85,
            populationSize = N,
        ),
        Evolutionary.Options(rng = rng, store_trace = true)
    )
    println("GA:TOUR(3):FLP:TPX (OneMax: 1/sum) => F: $(minimum(res)), C: $(Evolutionary.iterations(res))")
    @test sum(Evolutionary.minimizer(res)) >= N - 3
    @test abs(minimum(res)) >= N - 3
    @test Evolutionary.trace(res)[end].metadata["fitpop"][1] == minimum(res)

    # offspring that aren't crossed are copies of their parents: each child is
    # mutated once, and neither the parents nor the elite change
    m = GA(
        populationSize = 10, ɛ = 1, crossoverRate = 0.0, mutationRate = 1.0,
        selection = tournament(2), crossover = TPX, mutation = flip
    )
    opts = Evolutionary.Options(rng = rng)
    population = [falses(N) for _ in 1:m.populationSize]
    objfun = Evolutionary.EvolutionaryObjective(x -> Float64(sum(x)), first(population))
    state = Evolutionary.initial_state(m, opts, objfun, population)
    parents = copy(population)
    Evolutionary.update_state!(objfun, Evolutionary.NoConstraints(), state, population, m, opts, 1)
    @test allunique(objectid.(population))
    @test all(count(child) == 1 for child in population[1:m.populationSize])
    @test count(last(population)) == 0
    @test all(count(parent) == 0 for parent in parents)
    @test state.fitness == 0

end
