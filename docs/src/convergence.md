# Convergence

Convergence criteria can be specified for each optimization algorithm by setting
`metrics` parameter with an array of [`ConvergenceMetric`](@ref)`-derived objects,
e.g.

```julia
Evolutionary.optimize( f, x0, CMAES(metrics=[Evolutionary.AbsDiff(1e-5)]) )
```

Use `Evolutionary.converged(result)` to check convergence of the optimization
algorithm. It is possible to access a minimizer using `Evolutionary.minimizer(result)`
even if all convergence flags are `false`. This means that the user has to be
a bit careful when using the output from the solvers. It is advised to include
checks for convergence if the minimizer or minimum is used to carry out further
calculations.

## Convergence Metrics

```@docs
Evolutionary.AbsDiff
Evolutionary.RelDiff
Evolutionary.GD
Evolutionary.Never
```

## Auxiliary Functions

```@docs
Evolutionary.gd
Evolutionary.igd
Evolutionary.spread
```

