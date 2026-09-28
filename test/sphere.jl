using Evolutionary
using Test
using Random
using StableRNGs

@testset "Sphere" begin

    rng = StableRNG(42)

    # Objective function
    sphere(x::AbstractVector) = sum(x .* x)

    # Parameters
    N = 30
    P = 25
    initial = ones(N)

    function Evolutionary.trace!(record::Dict{String, Any}, objfun, state, population, method::ES, options)
        record["fitpop"] = state.fitness
        record["σ"] = strategy(state).σ
    end

    function Evolutionary.terminate(state::Evolutionary.ESState)
        strategy(state).σ < 1.0e-10
    end

    opts = Evolutionary.Options(show_trace = false, iterations = 1000, rng = rng)

    # Testing: (μ/μ_I, λ)-σ-Self-Adaptation-ES
    # with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
    result = Evolutionary.optimize(
        sphere,
        initial,
        ES(
            initStrategy = IsotropicStrategy(N),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            selection = :comma,
            μ = 3, λ = P
        ), opts
    )
    # show(result)
    println("(3/3,$(P))-σ-SA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test minimum(result) ≈ 0.0 atol = 1.0e-3
    @test sum(x -> x .^ 2, Evolutionary.minimizer(result)) ≈ 0.0 atol = 1.0e-3
    @test length(Evolutionary.minimizer(result)) == N

    # disable custom functions
    Evolutionary.terminate(state::Evolutionary.ESState) = false
    Evolutionary.trace!(record::Dict{String, Any}, objfun, state, population, method::ES, options) = ()

    # Testing: GA
    Random.seed!(rng, 42)
    result = Evolutionary.optimize(
        sphere,
        initial,
        GA(
            populationSize = 4P,
            mutationRate = 0.15,
            ɛ = 0.1,
            selection = susinv,
            crossover = IC(0.25),
            mutation = BGA(fill(0.5, N)),
        ), Evolutionary.Options(rng = rng)
    )
    # show(result)
    println("GA:INTER:DOMRNG:(N=$(N), P=$(P)) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test minimum(result) ≈ 0.0 atol = 1.0e-2
    @test sum(x -> x .^ 2, Evolutionary.minimizer(result)) ≈ 0.0 atol = 1.0e-2
    @test length(Evolutionary.minimizer(result)) == N

    # (μ+λ)-ES: the surviving parents are kept, best first
    m = ES(mutation = (x, s; kwargs...) -> x .- 1.0, μ = 2, ρ = 1, λ = 1, selection = :plus)
    opts = Evolutionary.Options(rng = rng)
    population = [[5.0], [0.0]]
    objfun = Evolutionary.EvolutionaryObjective(sphere, first(population))
    state = Evolutionary.initial_state(m, opts, objfun, population)
    Evolutionary.update_state!(objfun, Evolutionary.NoConstraints(), state, population, m, opts, 1)
    @test first(population) == [0.0]
    @test state.fitness == sphere.(population)
    @test issorted(state.fitness)

end
