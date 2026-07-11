# Evolution Strategies

```@docs
ES
```

## Description

The [Evolution Strategy](https://en.wikipedia.org/wiki/Evolution_strategy)  is  is an optimization technique based on ideas of evolution.

Evolution strategies use natural problem-dependent representations, and primarily [Mutation](@ref) and [Selection](@ref), as search operators.

The canonical versions of the ES are denoted by $(\sigma/\rho,\lambda)-ES$ and **(μ/ρ+λ)-ES**, respectively. Here $\mu$ denotes the number of parents, $\rho \leq \mu$ the mixing number (i.e., the number of parents involved in the procreation of an offspring), and $\lambda$ the number of offspring. The parents are deterministically selected (i.e., deterministic survivor selection) from the (multi-)set of either the offspring, referred to as **comma**-selection ($\mu < \lambda$ must hold), or both the parents and offspring, referred to as **plus**-selection [^1].

## Strategies

The evolution strategy algorithm provides, for every the optimized object parameter vector $x$, a set of strategy parameters $s$. The strategy is used to create an offspring $x^\prime$ is generated from the population individual $x$ on every iteration of the algorithm by applying a mutation operation:

$$x^\prime = mutation(x, s)$$

A strategy `s` usually has a parameter, e.g. $\sigma$, that controls the strength of the object parameter mutation. For example, if the mutation operation is [`gaussian`](@ref) then the $\sigma$ is simply the standard deviation of the normally distributed random component.

List of ES strategies:

```@docs
AbstractStrategy
NoStrategy
IsotropicStrategy
IsotropicStrategy(::Integer)
AnisotropicStrategy
AnisotropicStrategy(::Integer)
strategy
```

See [Mutation](@ref) section for strategy mutation operations.

## References

[^1]: [http://www.scholarpedia.org/article/Evolution_strategies](http://www.scholarpedia.org/article/Evolution_strategies)
