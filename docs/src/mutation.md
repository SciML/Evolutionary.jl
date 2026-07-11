# Mutation

In genetic algorithms and evolutionary computation, **mutation** is a genetic operator used to maintain a diversity from one generation of a population to the next. It is analogous to biological mutation. Mutation alters one or more gene values in a chromosome from its initial state.
The purpose of mutation is to introduce diversity into the sampled population.

## Mutation Interface

All mutation operations have following call interface `mutation(individual)` where `individual` is the member of population. The `mutation` function returns an **in-place mutated** individual.

## Evolutionary Strategy

See [Strategies](@ref) section for detailed description of ES strategies.

List of ES mutation operations:

```@docs
Evolutionary.nop(::AbstractVector, ::AbstractStrategy)
gaussian(::AbstractVector, ::IsotropicStrategy)
gaussian(::AbstractVector, ::AnisotropicStrategy)
cauchy(::AbstractVector, ::IsotropicStrategy)
```

List of ES strategy mutation operations:

```@docs
Evolutionary.nop(::AbstractStrategy)
gaussian(::IsotropicStrategy)
gaussian(::AnisotropicStrategy)
```


## Genetic Algorithm

The default (no-op) mutation operator simply returns the individual unchanged.

```@docs
Evolutionary.genop(recombinant)
mutationwrapper(::Function)
```

### Binary Mutations

```@docs
flip
bitinversion
```

### Real-valued Mutations

```@docs
uniform(::Real)
gaussian
BGA
PM
MIPM
PLM
```

### Combinatorial Mutations

*Note: The combinatorial mutation operations are applicable to binary vectors.*

```@docs
inversion
insertion
swap2
scramble
shifting
Evolutionary.replace
```

## Differential Evolution

```@docs
Evolutionary.differentiation
```

## Genetic Programming

```@docs
subtree
point
hoist
shrink
```
