# Development

```@meta
CurrentModule = Evolutionary
```

If you are contributing a new algorithm to this package, you need to know an internal API which allows to add a new algorithm without considerable changes to overall structure of the package.

## Adding an algorithm

If you're contributing a new algorithm, you shouldn't need to touch any of the code in src/api/optimize.jl. You should rather add a file named (solver is the name of the solver) `algo.jl` in src, and make sure that you define an optimizer **parameters** and **state** types, `initial_population` that initializes a population of `individual` objects, a state type that holds all variables that are (re)used throughout the iterative procedure, an `initial_state` that initializes such a state, and an `update_state!` method that does the actual work.


### Algorithm

Every optimization algorithm have to implement an algorithm parameters type derived from
 `AbstractOptimizer` type,  e.g. `struct Algo <: AbstractOptimizer end`, with appropriate fields, a default constructor with a keyword for each field.

Function `initial_state` returns an initial state for the algorithm, see [State](#state) section.
Function `update_state!` returns a `Bool` value. If the state update is *successfully* completed then the function returns `false`, otherwise `true`.

```@docs
AbstractOptimizer
initial_state
```

### State

Every optimization algorithm have to implement a state type derived from `AbstractOptimizerState` type, e.g. `struct AlgoState <: AbstractOptimizerState end`. All derived types should implement `value` and `minimizer` functions

```@docs
AbstractOptimizerState
value(::AbstractOptimizerState)
minimizer(::AbstractOptimizerState)
terminate(::AbstractOptimizerState)
```

### Reexported Objective API

Evolutionary extends and reexports the following objective-access functions for use
with `EvolutionaryObjective` values when implementing an optimizer state or evaluation
strategy.

```@docs
Evolutionary.value
Evolutionary.value!
Evolutionary.value!!
Evolutionary.f_calls
```

### Population

The evolutionary algorithms require a collection of individuals, **population**, which the algorithm constantly modifies. The population collection type must be derived from the `AbstractVector` type. Function `initial_population` is used for implementing a strategy of population collection initialization.

The `initial_population` must accept two parameters:
- `method`, an algorithm object derived from `AbstractOptimizer` type
- `individual`, a description of an individual template used to create the population


Following population initialization strategies are available:

```@docs
initial_population
```

### Constraints

All constraints derived from the `AbstractConstraints` abstract type.
For the derived type, the [`constraints interface functions`](@ref cons_aux_funcs)
have to be implemented.
If the derived type wraps a `ConstraintBounds` object, the [`bounds`](@ref) method
should be implemented.

Following auxiliary functions are available for every derived type of `AbstractConstraints`.

```@docs
isfeasible(::AbstractConstraints, x)
```

Package provides following additional constrains implementations.

```@docs
Evolutionary.NoConstraints
MixedTypePenaltyConstraints
```

### Objective

Internally, the objective function is wrapped into [`EvolutionaryObjective`](@ref) type object.

```@docs
Evolutionary.EvolutionaryObjective
Evolutionary.EvolutionaryObjective(f, x::AbstractArray)
Evolutionary.EvolutionaryObjective(f, x::Expr)
Evolutionary.ismultiobjective
```

### Parallelization

For additional modes of parallelization of the objective function evaluation, add overrides of the `value!` function.
By default, the fitness of the population is calculated by the following function:

```julia
function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:serial}}, fitness, population::AbstractVector{IT}) where {IT}
    n = length(xs)
    for i in 1:n
        F[i] = value(obj, xs[i])
    end
    F
end
```

The first symbolic value type parameter, `:serial`, corresponds to the default value
of the `parallelization` of the [`Options`](@ref) object.
Any additional overrides with different value type parameters will be triggered
by specifying a corresponded value type symbol in the `Options.parallelization` field.
A multi-threaded override of the above evaluation is provided.

## Convergence Metrics

In order to add a new convergence metric, create a new type derived from `ConvergenceMetric` abstract type.
In addition, every convergence metric class should implement following interface:

```@docs
Evolutionary.ConvergenceMetric
Evolutionary.description
Evolutionary.converged
Evolutionary.assess!
Evolutionary.diff
Evolutionary.tolerance
```

**Note:** If the class contains the field `Δ`, which holds the current difference
between last consecutive convergence function evaluations, then the provided
`Evolutinary.diff` function will work correctly.
Similarly, if the class has the field `tol`, which holds a tolerance value,
the `Evolutionary.tolerance` method will work correctly.
