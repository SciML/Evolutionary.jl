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
export uniformbin, exponential, singlepoint, twopoint, domainrange, waverage,
    intermediate, line, discrete

"""
    uniformbin(args...; kwargs...)

Deprecated alias for [`BINX`](@ref).
"""
function uniformbin(args...; kwargs...)
    Base.depwarn("`uniformbin` is deprecated, use `BINX` instead.", :uniformbin)
    return BINX(args...; kwargs...)
end

"""
    exponential(args...; kwargs...)

Deprecated alias for [`EXPX`](@ref).
"""
function exponential(args...; kwargs...)
    Base.depwarn("`exponential` is deprecated, use `EXPX` instead.", :exponential)
    return EXPX(args...; kwargs...)
end

"""
    singlepoint(args...; kwargs...)

Deprecated alias for [`SPX`](@ref).
"""
function singlepoint(args...; kwargs...)
    Base.depwarn("`singlepoint` is deprecated, use `SPX` instead.", :singlepoint)
    return SPX(args...; kwargs...)
end

"""
    twopoint(args...; kwargs...)

Deprecated alias for [`TPX`](@ref).
"""
function twopoint(args...; kwargs...)
    Base.depwarn("`twopoint` is deprecated, use `TPX` instead.", :twopoint)
    return TPX(args...; kwargs...)
end

"""
    domainrange(args...; kwargs...)

Deprecated alias for [`BGA`](@ref).
"""
function domainrange(args...; kwargs...)
    Base.depwarn("`domainrange` is deprecated, use `BGA` instead.", :domainrange)
    return BGA(args...; kwargs...)
end

"""
    waverage(args...; kwargs...)

Deprecated alias for [`WAX`](@ref).
"""
function waverage(args...; kwargs...)
    Base.depwarn("`waverage` is deprecated, use `WAX` instead.", :waverage)
    return WAX(args...; kwargs...)
end

"""
    intermediate(args...; kwargs...)

Deprecated alias for [`IC`](@ref).
"""
function intermediate(args...; kwargs...)
    Base.depwarn("`intermediate` is deprecated, use `IC` instead.", :intermediate)
    return IC(args...; kwargs...)
end

"""
    line(args...; kwargs...)

Deprecated alias for [`LC`](@ref).
"""
function line(args...; kwargs...)
    Base.depwarn("`line` is deprecated, use `LC` instead.", :line)
    return LC(args...; kwargs...)
end

"""
    discrete(args...; kwargs...)

Deprecated alias for [`DC`](@ref).
"""
function discrete(args...; kwargs...)
    Base.depwarn("`discrete` is deprecated, use `DC` instead.", :discrete)
    return DC(args...; kwargs...)
end

end
