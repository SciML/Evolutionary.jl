"""
    Terminal

Allowed terminal element types for tree-based genetic programming expressions.
"""
const Terminal = Union{Symbol, Real, Function}

"""
Implementation of Koza-type (tree-based) Genetic Programming

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `terminals`: A dictionary of terminals with their their corresponding dimensionality
    - This dictionary contains (`Terminal`, `Int`) pairs
    - The terminals can be any symbols (variables), constant values, or 0-arity functions.
- `functions`: A collection of functions with their corresponding arity.
    - This dictionary contains (`Function`, `Int`) pairs
- `initialization`: A strategy for population initialization (default: `:grow`)
    - Possible values: `:grow` and `:full`
- `mindepth`: Minimal depth of the expression (default: `0`)
- `maxdepth`: Maximal depth of the expression (default: `3`)
- `mutation`: A mutation function (default: [`crosstree`](@ref))
- `crossover`: A crossover function (default: [`subtree`](@ref))
- `simplify`: An expression simplification function (default: `:nothing`)
- Use `mutation` and `crossover` parameters to specify GP-related mutation operation.
- Use `selection` parameter to specify the offspring selection procedure
"""
@kwdef struct TreeGP <: AbstractOptimizer
    populationSize::Integer = 50
    terminals::Dict{Terminal, Int} = Dict(:x => 1, rand => 1)
    functions::Dict{Function, Int} = Dict(f => 2 for f in [+, -, *, pdiv])
    mindepth::Int = 0
    maxdepth::Int = 3
    crossover::Function = crosstree
    mutation::Function = subtree
    selection::Function = tournament(2)
    crossoverRate::Real = 0.9
    mutationRate::Real = 0.1
    initialization::Symbol = :grow
    simplify::Union{Nothing, Function} = nothing
    metrics::ConvergenceMetrics = ConvergenceMetric[AbsDiff(1.0e-5)]
end
function TreeGP(pop::Integer, term::Vector{Terminal}, func::Vector{Function}; kwargs...)
    terminals = Dict(t => 1 for t in term)
    functions = Dict(f => 2 for f in func)
    return TreeGP(; populationSize = pop, terminals = terminals, functions = functions, kwargs...)
end

population_size(method::TreeGP) = method.populationSize
default_options(method::TreeGP) = (iterations = 1000,)
terminals(m::TreeGP) = Symbol[t for t in keys(m.terminals) if isa(t, Symbol)] |> sort!
function summary(m::TreeGP)
    par = join(terminals(m), ",")
    return "TreeGP[P=$(m.populationSize),Parameter[$(par)],$(keys(m.functions))]"
end
show(io::IO, m::TreeGP) = print(io, summary(m))

"""
    randterm(t::TreeGP)

Returns a random terminal given the specification from the `TreeGP` object `t`.
"""
function randterm(rng::AbstractRNG, t::TreeGP)
    term = rand(rng, keys(t.terminals))
    return if isa(term, Symbol) || isa(term, Real)
        term
    elseif isa(term, Function)
        term(rng) # terminal functions must accept RNG as an argument
    else
        # Code shouldn't reach branch but left as a catchall
        dim = t.terminals[term]
        dim == 1 ? rand(rng) : rand(rng, dim)
    end
end
randterm(t::TreeGP) = randterm(default_rng(), t)

"""
    rand(t::TreeGP, maxdepth=2; mindepth=maxdepth-1)::Expr

Create a random expression tree given the specification from the `TreeGP` object `t`.
"""
function rand(rng::AbstractRNG, t::TreeGP, maxdepth::Int = 2; mindepth::Int = maxdepth - 1)
    @assert maxdepth > mindepth "`maxdepth` must be larger then `mindepth`"
    tl = length(t.terminals)
    fl = length(t.functions)
    # generate a root of a subtree
    root = if (maxdepth == 0  || (t.initialization == :grow && rand(rng) < tl / (tl + fl))) && mindepth <= 0
        randterm(rng, t)
    else
        rand(rng, keys(t.functions))
    end
    # if the root element is a function add subtrees as its parameters
    if isa(root, Function)
        args = Any[]
        for i in 1:t.functions[root]
            arg = rand(rng, t, maxdepth - 1, mindepth = mindepth - 1)
            push!(args, arg)
        end
        Expr(:call, root, args...)
    else
        return root
    end
end
rand(t::TreeGP, maxdepth::Int = 2; kwargs...) =
    rand(default_rng(), t, maxdepth; kwargs...)

"""
    initial_population(m::TreeGP, expr::{Expr,Nothing}=nothing)

Initialize a random population of expressions derived from `expr`.
"""
function initial_population(
        m::TreeGP, expr::Union{Expr, Nothing} = nothing;
        rng::AbstractRNG = default_rng()
    )
    n = population_size(m)
    if isnothing(expr)
        return [ rand(rng, m, m.maxdepth, mindepth = m.mindepth) for i in 1:n ]
    else
        return [ deepcopy(expr) for i in 1:n ]
    end
end

mutable struct GPState{T, IT, OT <: AbstractOptimizer} <: AbstractOptimizerState
    optimizer::OT
    state::GAState{T, IT}
end
value(s::GPState) = s.state.fitness
minimizer(s::GPState) = s.state.fittest

"""Initialization of GP algorithm state"""
function initial_state(method::TreeGP, options, objfun, population)
    ga = GA(
        populationSize = method.populationSize,
        crossover = method.crossover,
        mutation = method.mutation(method),
        selection = method.selection,
        crossoverRate = method.crossoverRate,
        mutationRate = method.mutationRate
    )
    gas = initial_state(ga, options, objfun, population)
    return GPState(ga, gas)
end

function update_state!(objfun, constraints, state::GPState, population::AbstractVector{IT}, method, options, itr) where {IT}
    # perform GA step
    res = update_state!(objfun, constraints, state.state, population, state.optimizer, options, itr)
    # simplify expressions
    if method.simplify !== nothing
        for i in 1:length(population)
            method.simplify(population[i])
        end
    end
    return res
end

# Custom optimization call
optimize(f, mthd::TreeGP, options::Options = Options(; default_options(mthd)...)) =
    optimize(f, NoConstraints(), mthd, initial_population(mthd, rng = options.rng), options)
