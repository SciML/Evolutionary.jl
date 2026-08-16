using Documenter, Evolutionary

makedocs(
    modules = [Evolutionary],
    clean = true,
    sitename = "Evolutionary.jl",
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Constraints" => "constraints.md",
        "Convergence" => "convergence.md",
        "Algorithms" => [
            "Genetic Algorithm" => "ga.md",
            "Differential Evolution" => "de.md",
            "Evolution Strategy" => "es.md",
            "CMA-ES" => "cmaes.md",
            "Genetic Programming" => "gp.md",
            "Multi-objective EA" => "moea.md",
        ],
        "Operations" => [
            "Selection" => "selection.md",
            "Crossover" => "crossover.md",
            "Mutation" => "mutation.md",
        ],
        "Development" => "dev.md",
    ]
)

deploydocs(repo = "github.com/SciML/Evolutionary.jl.git")
