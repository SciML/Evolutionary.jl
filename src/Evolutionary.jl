module Evolutionary
using LinearAlgebra: Diagonal, Symmetric, diag, diagm, eigen!, norm
using Statistics: mean
using Base: @kwdef
using UnPack: @unpack
using StackViews: StackView
import PrecompileTools: @compile_workload, @setup_workload
using Random: AbstractRNG, default_rng, randperm, shuffle, randn!
using NLSolversBase: NLSolversBase, AbstractObjective, ConstraintBounds,
    AbstractConstraints
import NLSolversBase: f_calls, value, value!, value!!
import Base: show, copy, minimum, summary, getproperty, rand, length

export AbstractStrategy, strategy, mutationwrapper,
    IsotropicStrategy, AnisotropicStrategy, NoStrategy,
    isfeasible, BoxConstraints, apply!, penalty, penalty!, bounds, constraint_values,
    PenaltyConstraints, WorstFitnessConstraints, MixedTypePenaltyConstraints,
    EvolutionaryObjective, ismultiobjective, ConvergenceMetric, default_options,
    # ES mutations
    gaussian, cauchy,
    # GA mutations
    flip, bitinversion, uniform, BGA, inversion, insertion, swap2, scramble,
    shifting, PM, MIPM, PLM, replacement,
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
    ES, CMAES, GA, DE, TreeGP, NSGA2

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
function uniformbin(Cr::Real = 0.5)
    Base.depwarn("`uniformbin` is deprecated, use `BINX` instead.", :uniformbin)
    return BINX(Cr)
end

"""
    exponential(args...; kwargs...)

Deprecated alias for [`EXPX`](@ref).
"""
function exponential(Cr::Real = 0.5)
    Base.depwarn("`exponential` is deprecated, use `EXPX` instead.", :exponential)
    return EXPX(Cr)
end

"""
    singlepoint(args...; kwargs...)

Deprecated alias for [`SPX`](@ref).
"""
function singlepoint(v1::T, v2::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    Base.depwarn("`singlepoint` is deprecated, use `SPX` instead.", :singlepoint)
    return SPX(v1, v2; rng)
end

"""
    twopoint(args...; kwargs...)

Deprecated alias for [`TPX`](@ref).
"""
function twopoint(v1::T, v2::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    Base.depwarn("`twopoint` is deprecated, use `TPX` instead.", :twopoint)
    return TPX(v1, v2; rng)
end

"""
    domainrange(args...; kwargs...)

Deprecated alias for [`BGA`](@ref).
"""
function domainrange(valrange::Vector, m::Int = 20)
    Base.depwarn("`domainrange` is deprecated, use `BGA` instead.", :domainrange)
    return BGA(valrange, m)
end

"""
    waverage(args...; kwargs...)

Deprecated alias for [`WAX`](@ref).
"""
function waverage(weights::Vector{<:Real})
    Base.depwarn("`waverage` is deprecated, use `WAX` instead.", :waverage)
    return WAX(weights)
end

"""
    intermediate(args...; kwargs...)

Deprecated alias for [`IC`](@ref).
"""
function intermediate(d::Real = 0.0)
    Base.depwarn("`intermediate` is deprecated, use `IC` instead.", :intermediate)
    return IC(d)
end

"""
    line(args...; kwargs...)

Deprecated alias for [`LC`](@ref).
"""
function line(d::Real = 0.0)
    Base.depwarn("`line` is deprecated, use `LC` instead.", :line)
    return LC(d)
end

"""
    discrete(args...; kwargs...)

Deprecated alias for [`DC`](@ref).
"""
function discrete(v1::T, v2::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    Base.depwarn("`discrete` is deprecated, use `DC` instead.", :discrete)
    return DC(v1, v2; rng)
end

@setup_workload begin
    @compile_workload begin
        objective(x) = sum(abs2, x)
        initial = () -> [0.25, -0.5]
        options = Options(iterations = 2)
        optimize(objective, initial, GA(populationSize = 4), options)
        optimize(objective, initial, DE(populationSize = 4), options)
    end
end

end
