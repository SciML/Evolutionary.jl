using Evolutionary, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Rastrigin-like multimodal objective
function rastrigin(x)
    return 10 * length(x) + sum(abs2, x) - 10 * sum(cos.(2π .* x))
end
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

x0 = rand(rng, 10)
bounds = Evolutionary.BoxConstraints(fill(-5.0, 10), fill(5.0, 10))

opts(iter) = Evolutionary.Options(; iterations = iter, abstol = 1.0e-12)

# =============================================================================
# Optimizers
# =============================================================================

SUITE["optimize"] = BenchmarkGroup()

SUITE["optimize"]["ga"] = @benchmarkable Evolutionary.optimize(
    $rastrigin, $bounds, $x0, GA(),
    $(Evolutionary.Options(; iterations = 200))
)
SUITE["optimize"]["cmaes"] = @benchmarkable Evolutionary.optimize(
    $rastrigin, $x0, CMAES(; μ = 10, λ = 20),
    $(Evolutionary.Options(; iterations = 200))
)
SUITE["optimize"]["de"] = @benchmarkable Evolutionary.optimize(
    $rastrigin, $bounds, DE(; n = 20),
    $(Evolutionary.Options(; iterations = 200))
)
SUITE["optimize"]["es"] = @benchmarkable Evolutionary.optimize(
    $rosenbrock, [0.5, 0.5], ES(; μ = 10, ρ = 1, λ = 20),
    $(Evolutionary.Options(; iterations = 200))
)
