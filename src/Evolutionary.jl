module Evolutionary
using LinearAlgebra: Diagonal, Symmetric, diag, diagm, eigen!, norm
using Statistics: mean
using Base: @kwdef
using UnPack: @unpack
using StackViews: StackView
using Random: AbstractRNG, default_rng, randperm, shuffle, randn!
using NLSolversBase: NLSolversBase, AbstractObjective, ConstraintBounds,
    AbstractConstraints, nconstraints_x, nconstraints
import NLSolversBase: f_calls, value, value!, value!!
import Base: show, copy, minimum, summary, getproperty, rand, getindex, length,
    copyto!, setindex!, replace

export AbstractStrategy, strategy, mutationwrapper,
    IsotropicStrategy, AnisotropicStrategy, NoStrategy,
    isfeasible, BoxConstraints, apply!, penalty, penalty!, bounds,
    PenaltyConstraints, WorstFitnessConstraints, MixedTypePenaltyConstraints,
    EvolutionaryObjective, ismultiobjective, ConvergenceMetric, default_options,
    # ES mutations
    gaussian, cauchy,
    # GA mutations
    flip, bitinversion, uniform, BGA, inversion, insertion, swap2, scramble,
    shifting, PM, MIPM, PLM, replace,
    # ES recombinations
    average, marriage,
    # GA recombinations
    SPX, TPX, UX, SHFX,
    DC, AX, WAX, IC, LC, HX, LX, MILX, SBX,
    PMX, OX1, CX, OX2, POS,
    BSX, SSX,
    # GA selections
    ranklinear, uniformranking, roulette, rouletteinv, sus, susinv,
    tournament, truncation,
    # DE selections
    random, permutation, randomoffset, best,
    # DE recombinations
    BINX, EXPX,
    # GP exports
    Terminal, subtree, point, hoist, shrink,
    # Optimization methods
    ES, CMAES, GA, DE, TreeGP, NSGA2,
    # re-export
    value, value!, value!!, f_calls

# optimize API
include("api/types.jl")
include("api/objective.jl")
include("api/results.jl")
include("api/termination.jl")
include("api/utilities.jl")
include("api/constraints.jl")
include("api/optimize.jl")
include("api/expressions.jl")
include("api/moea.jl")

# Evolution Strategy
include("es.jl")
include("cmaes.jl")

# Genetic Algorithms
include("ga.jl")
include("nsga2.jl")

# Differential Evolution
include("de.jl")

# Genetic Programming
include("api/protected.jl")
include("gp.jl")

# ES & GA recombination functions
include("recombinations.jl")

# ES & GA mutation functions
include("mutations.jl")

# GA selection functions
include("selections.jl")

@deprecate uniform(v1, v2) UX(v1, v2)
@deprecate uniformbin BINX
@deprecate exponential EXPX
@deprecate singlepoint SPX
@deprecate twopoint TPX
@deprecate domainrange BGA
@deprecate waverage WAX
@deprecate intermediate IC
@deprecate line LC
@deprecate discrete DC

end
