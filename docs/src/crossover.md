# Crossover

In genetic algorithms and evolutionary computation, **crossover**, also called **recombination**, is a genetic operator used to combine the genetic information of two parents to generate new offspring.

## Recombination Interface

All recombination operations have following call interface: `recombination(i1, i2)` where `i1` and `i2` are the same type individuals that involved in recombination to produce an offspring. The recombination function returns pair of recombined individuals.

**Note:** Some of the selection algorithms implemented as function closures, in order to provide additional parameters for the specified above recombination interface.

## Operations

### ES Crossovers

List of the ES strategy recombination operations:

```@docs
average(::Vector{<:AbstractStrategy})
```

List of the ES population recombination operations:

```@docs
average(population::Vector{T}) where {T <: AbstractVector}
marriage
```

### Binary Crossovers

```@docs
SPX
TPX
SHFX
UX
BINX
EXPX
BSX
singlepoint
twopoint
uniformbin
exponential
```

### Real-valued Crossovers

```@docs
Evolutionary.genop(::T,::T) where {T<:AbstractVector}
DC
AX
WAX
IC
LC
HX
LX
MILX
SBX
discrete
waverage
intermediate
line
domainrange
```

### Combinatorial Crossovers

```@docs
PMX
OX1
CX
OX2
POS
SSX
```

### Tree (expression) Crossovers

```@docs
Evolutionary.crosstree
```
