using Evolutionary
using Test
using Random
using StableRNGs

@testset "Rastrigin" begin

    rng = StableRNG(42)

    function test_result(result::Evolutionary.EvolutionaryOptimizationResults, N::Int, tol::Float64)
        fitness = minimum(result)
        extremum = Evolutionary.minimizer(result)
        if round(fitness) == 0
            @test extremum ≈ zeros(N) atol = tol
            @test fitness ≈ 0.0 atol = tol
        else
            # @warn("Found local minimum!!!")
            @test sum(abs, extremum) < N
        end
    end

    # Objective function
    function rastrigin(x::AbstractVector{T}) where {T <: AbstractFloat}
        n = length(x)
        return 10n + sum([ x[i]^2 - 10cos(convert(T, 2π * x[i])) for i in 1:n ])
    end

    # Parameters
    N = 3
    P = 100
    initState = () -> rand(rng, N)

    # Testing: (μ/μ_I,λ)-σ-Self-Adaptation-ES
    # with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))
    Random.seed!(rng, 42)
    m = ES(mu = 15, lambda = P)
    @test m.μ == 15
    @test m.λ == P
    opts = Evolutionary.Options(iterations = 1000, rng = rng)
    result = Evolutionary.optimize(
        rastrigin,
        initState,
        ES(
            initStrategy = AnisotropicStrategy(N),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            selection = :comma,
            μ = 15, λ = P
        ), opts
    )
    println("(15,$(P))-σ-SA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1.0e-1)

    # Testing: CMA-ES
    Random.seed!(rng, 42)
    opts = Evolutionary.Options(rng = rng)
    result = Evolutionary.optimize(rastrigin, initState, CMAES(lambda = P), opts)
    println("($(P >> 1),$P)-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1.0e-1)

    # Testing: GA
    m = GA(epsilon = 10.0)
    @test m.ɛ == 10
    opts = Evolutionary.Options(iterations = 1000, rng = rng, successive_f_tol = 25)
    selections = [:roulette => rouletteinv, :sus => susinv, :tourn => tournament(3)]
    crossovers = [:discrete => DC, :intermediate0 => IC(0.0), :intermediate0_5 => IC(0.5), :line => LC(0.1), :avg => AX, :heuristic => HX, :laplace => LX(), :simbin => SBX()]
    mutations = [:domrng0_5 => BGA(fill(0.5, N)), :gaussian => gaussian(), :plm => PLM()]

    @testset "GA settings" for (sn, ss) in selections, (xn, xovr) in crossovers, (mn, ms) in mutations
        (xn ∈ [:line, :discrete]) && mn == :gaussian && continue # bad combination
        Random.seed!(rng, 4)
        result = Evolutionary.optimize(
            rastrigin, initState,
            GA(
                populationSize = P,
                selection = ss,
                crossover = xovr,
                mutation = ms
            ), opts
        )
        println("GA:$(sn):$(xn):$(mn)(N=$(N),P=$(P),x=.8,μ=.1) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        test_result(result, N, 0.2)
    end

    # Testing: DE
    selections = [:rand => random, :perm => permutation, :rndoff => randomoffset, :best => best]
    mutations = [:exp => EXPX(0.5), :bin => BINX(0.5)]
    opts = Evolutionary.Options(rng = rng, successive_f_tol = 25)
    @testset "DE settings" for (sn, ss) in selections, (mn, ms) in mutations, n in 1:2
        Random.seed!(rng, 1)
        result = Evolutionary.optimize(
            rastrigin, initState,
            DE(
                populationSize = P,
                n = n,
                selection = ss,
                recombination = ms,
                F = 0.9
            ),
            opts
        )
        println("DE/$sn/$n/$mn(F=0.9,Cr=0.5) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        #println(Evolutionary.minimizer(result))
        test_result(result, N, 1.0e-4)
    end

end
