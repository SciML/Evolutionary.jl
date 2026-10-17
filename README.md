# Evolutionary

A Julia package for [evolutionary](http://www.scholarpedia.org/article/Evolution_strategies) & [genetic](http://en.wikipedia.org/wiki/Genetic_algorithm) algorithms.

| **Documentation** | **Build Status** | **References** |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] [![][coverage-img]][coverage-url] | [![][doi-img]][doi-url]


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Evolutionary
```

## Algorithms

- (μ/ρ(+/,)λ)-SA-ES (ES)
- (μ/μ<sub>I</sub>,λ)-CMA-ES (CMAES)
- Genetic Algorithms (GA)
  - Multi-objective: NSGA-II (NSGA2)
- Differential Evolution (DE)
- Genetic Programming (TreeGP)

## Operators

- Mutations
  - ES
    - (an)isotropic Gaussian
    - (an)isotropic Cauchy
  - GA
    - binary
      - flip
      - inversion
    - real valued
      - uniform
      - Gaussian
      - BGA (domain range)
      - (MI)PM (power)
      - PLM (polynomial)
    - combinatorial
      - inversion
      - insertion
      - swap2
      - scramble
      - shifting

- Recombinations
  - average
  - marriage

- Crossovers
  - binary
    - SPX (single point)
    - TPX (two point)
    - SHFX (shuffle)
    - UX (uniform)
    - BINX (binary)
    - EXPX (exponential)
    - BSX (binary subset)
  - real valued
    - DC (discrete)
    - AX (average)
    - WAX (weighted average)
    - IC (intermediate)
    - LC (line)
    - HX (heuristic)
    - (MI)LX (Laplace)
    - SBX (simulated binary)
  - combinatorial
    - PMX (partially-mapped)
    - OX1 (order)
    - OX2 (order-based)
    - CX (cycle)
    - POS (position-based)

- Selections
  - rank-based fitness assignment
  - (μ, λ)-uniform ranking
  - roulette (fitness proportionate selection, FSP)
  - stochastic universal sampling (SUS)
  - tournament
  - truncation


[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://docs.sciml.ai/Evolutionary/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://docs.sciml.ai/Evolutionary/stable/

[travis-img]: https://travis-ci.org/wildart/Evolutionary.jl.svg?branch=master
[travis-url]: https://travis-ci.org/wildart/Evolutionary.jl

[CI-img]: https://github.com/wildart/Evolutionary.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/wildart/Evolutionary.jl/actions/workflows/CI.yml

[coverage-img]: https://img.shields.io/coveralls/wildart/Evolutionary.jl.svg
[coverage-url]: https://coveralls.io/r/wildart/Evolutionary.jl?branch=master

[issues-url]: https://github.com/wildart/Evolutionary.jl/issues

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.5110647.svg
[doi-url]: https://doi.org/10.5281/zenodo.5110647
