using Evolutionary
using Random
using Test

objective(x) = sum(abs2, x)
initial = () -> [0.25, -0.5]
options = Evolutionary.Options(iterations = 2, rng = MersenneTwister(1))

for algorithm in (GA(populationSize = 4), DE(populationSize = 4))
    result = Evolutionary.optimize(objective, initial, algorithm, options)
    @test isfinite(minimum(result))
    @test length(Evolutionary.minimizer(result)) == 2
end
