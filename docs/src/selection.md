# Selection

**Selection** is a genetic operator used in EAs for selecting potentially useful solutions from a population for later breeding.
The EAs are stochastic search methods using the concepts of Mendelian genetics and Darwinian evolution.
According to Darwin's evolution theory the best ones should survive and create new offspring.
There are many methods how to select the best individuals, for example roulette wheel selection, Boltzmann selection,
tournament selection, rank selection, steady state selection and some others.

## Selection Interface

All selection algorithms have following call interface `selection(fitness, N)` where `fitness` is the vector of population fitness values, of size ``M``, and ``N`` is the number of selected individuals. The selection function returns a vector of integer indexes of selected individuals, of size ``N`` with indexes in range ``[1,M]``.

**Note:** Some of the selection algorithms implemented as function closures, in order to provide additional parameters to the specified above selection interface.

## Genetic Algorithm

```@docs
ranklinear
uniformranking
roulette
rouletteinv
sus
susinv
truncation
tournament
```

## Differential Evolution

```@docs
random
permutation
randomoffset
best
```