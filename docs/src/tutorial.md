```@meta
CurrentModule = Evolutionary
```

```@setup result
using Evolutionary
```

## Optimization

To show how the **Evolutionary** package can be used, we minimize the
[Rosenbrock function](http://en.wikipedia.org/wiki/Rosenbrock_function),
a classical test problem for numerical optimization. We'll assume that you've already
installed the **Evolutionary** package using Julia's package manager.

### Objective Function Definition

First, we load **Evolutionary** and define the Rosenbrock function:

```julia
using Evolutionary
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 # Rosenbrock
```

There are various ways to define your objective function:

- For single-objective optimization, the objective function has to have one parameter and return a scalar value, e.g.

```julia
soofun(x) = x[1] + x[2]
```

- For multi-objective optimization, the objective function has to return an vector of values, e.g.

```julia
moofun(x) = [ x[1], x[2]+1 ]
```

- If multi-objective function has one parameter, the resulting value array will be copied. To reduce, additional data copy, the function can be defined with two parameters to perform in-place change of the result, e.g.

```julia
function moofun(F, x)
    F[1] = x
    F[2] = x +1
    F
end
```

### Perform Optimization

Once we've defined this function, we can find the minimizer (the input that minimizes the objective) and the minimum (the value of the objective at the minimizer) using any of our favorite optimization algorithms. With a function defined,
we just specify a form of an individual `x` of the population for an evolutionary algorithm, and call `optimize` with a starting individual `x0` and a particular optimization algorithm, e.g. [`CMAES()`](@ref):

```julia
x0 = [0.0, 0.0];
Evolutionary.optimize(f, x0, CMAES())
```

```@docs
Evolutionary.optimize
```

## Configurable options

There are several options that simply take on some default values if the user doesn't provide any.

### Algorithm options

There are different algorithms available in `Evolutionary`, and they are all listed below. Notice that the constructors are written without input here, but they generally take keywords to tweak the way they work. See the pages describing each solver for more detail.

- [`GA()`](@ref)
- [`ES()`](@ref)
- [`CMAES()`](@ref)
- [`DE()`](@ref)
- [`NSGA2()`](@ref)
- [`TreeGP()`](@ref)

### General options

In addition to the algorithm, you can alter the behavior of the optimization procedure by using the following `Options` keyword arguments:
```@docs
Options
```

We currently recommend the statically dispatched interface by using the `Evolutionary.Options` constructor:

```julia
res = Evolutionary.optimize(x->-sum(x),
                            BitVector(zeros(30)),
                            GA(selection=uniformranking(5),
                            mutation=flip, crossover=SPX),
                            Evolutionary.Options(iterations=10))
```

## Obtaining results

After we have our results in `res` object, we can use the API for getting optimization results. This consists of a collection of functions. They are not exported, so they have to be prefixed by `Evolutionary.`. Say we do the following optimization:

```@repl result
res = Evolutionary.optimize(x->-sum(x), BitVector(zeros(3)), GA())
```

You can inspect the result by using a collection of the auxiliary functions, e.g. the `minimizer` and `minimum` of the objective functions, which can be found using

```@repl result
Evolutionary.minimizer(res)
Evolutionary.minimum(res)
```

#### Complete list of functions

An `OptimizationResults` interface for representing an optimization result.

```@docs
OptimizationResults
summary(::OptimizationResults)
minimizer(::OptimizationResults)
minimum(::OptimizationResults)
iterations(::OptimizationResults)
iteration_limit_reached(::OptimizationResults)
trace(::OptimizationResults)
NLSolversBase.f_calls(::OptimizationResults)
abstol(::OptimizationResults)
reltol(::OptimizationResults)
```

An implementation of the result object for evolutionary optimizations.
```@docs
EvolutionaryOptimizationResults
converged(::EvolutionaryOptimizationResults)
```

## Trace

When `store_trace` and/or `show_trace` options are set to `true` in the [`Options`](@ref) object, an optimization trace is either captured and/or shown on the screen. By default, only the current state minimum value is displayed in the trace. In order to extend the trace record, you need to override the [`trace!`](@ref) function by providing specialized function behavior on one of specific parameters.

```@docs
trace!(::Dict{String,Any}, Any, Any, Any, Any, Any)
```

Commonly, you would define a specializations of a `state`, `population`, or `method` parameters of `trace!` function, e.g.

```julia
function trace!(record::Dict{String,Any}, objfun, state, population, method::CMAES, options)
    record["σ"] = state.σ
end
```

## Parallelization

If the objective function is heavily CPU-bound, it's possible to utilize multiple processors/threads to speed up computations.
To enable multi-threading evaluation of the objective function, set `parallelization` option to `:thread` in the [`Options`](@ref) object.
