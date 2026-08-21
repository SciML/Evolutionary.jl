using SafeTestsets
using Test
using SciMLTesting

# The functional suite is identical across the Core and CoreLTSPre groups; those
# groups exist only to split the version/OS matrix (see test/test_groups.toml).
# Both run the same set of files, so they share one body thunk; `all = ["Core"]`
# keeps the local default (GROUP unset => "All") running the suite exactly once.
function functional_suite()
    @safetestset "Piracy guard" begin
        using Evolutionary
        # Guard against accidental piracy from `import`
        @test Evolutionary.contains !== Base.contains
    end
    @safetestset "Types" include("types.jl")
    @safetestset "Evolutionary Objective" include("objective.jl")
    @safetestset "API" include("interface.jl")
    @safetestset "Precompile workload" include("precompile_workload.jl")
    @safetestset "Selections" include("selections.jl")
    @safetestset "Recombinations" include("recombinations.jl")
    @safetestset "Mutations" include("mutations.jl")
    @safetestset "Sphere" include("sphere.jl")
    @safetestset "Rosenbrock" include("rosenbrock.jl")
    @safetestset "Schwefel CMA-ES" include("schwefel.jl")
    @safetestset "Rastrigin" include("rastrigin.jl")
    @safetestset "n-Queens" include("n-queens.jl")
    @safetestset "Knapsack" include("knapsack.jl")
    @safetestset "OneMax" include("onemax.jl")
    @safetestset "Multi-objective EA" include("moea.jl")
    @safetestset "Regression" include("regression.jl")
    @safetestset "Genetic Programming" include("gp.jl")
    return nothing
end

run_tests(;
    core = functional_suite,
    groups = Dict("CoreLTSPre" => functional_suite),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    all = ["Core"],
)
