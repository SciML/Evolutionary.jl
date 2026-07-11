# Genetic Programming

```@meta
CurrentModule = Evolutionary
```

```@docs
TreeGP
Terminal
```

## Description

The [Genetic Programming](https://en.wikipedia.org/wiki/Genetic_programming) is a technique of evolving programs, starting from a population of unfit (usually random) programs, fit for a particular task by applying operations analogous to natural genetic processes to the population of programs. It is essentially a heuristic search technique, i.e. searching for an optimal or at least suitable program among the space of all programs[^1].

## Auxiliary Functions

```@docs
Evolutionary.Expression
Evolutionary.randterm
Evolutionary.simplify!
Base.rand(::AbstractRNG, ::TreeGP)
```

## Protected Functions

The following protected operators guard against domain errors (e.g. division by
zero, logarithm of zero) so that arbitrary expression trees can be evaluated safely.

```@docs
Evolutionary.pdiv
Evolutionary.aq
Evolutionary.pexp
Evolutionary.plog
Evolutionary.psqrt
Evolutionary.psin
Evolutionary.pcos
Evolutionary.ppow
Evolutionary.cond
```

## References

[^1]: John R. Koza, "Genetic Programming: On the Programming of Computers by Means of Natural Selection", MIT Press, 1992.
