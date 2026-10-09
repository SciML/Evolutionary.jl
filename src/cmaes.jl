"""
Covariance Matrix Adaptation Evolution Strategy Implementation: (μ/μ_{I,W},λ)-CMA-ES

The constructor takes following keyword arguments:

- `μ`/`mu` is the number of parents
- `λ`/`lambda` is the number of offspring
- `c_1` is a learning rate for the rank-one update of the covariance matrix update
- `c_c` is a learning rate for cumulation for the rank-one update of the covariance matrix
- `c_mu` is a learning rate for the rank-``\\mu`` update of the covariance matrix update
- `c_sigma` is a learning rate for the cumulation for the step-size control
- `c_m` is the learning rate for the mean update, ``c_m \\leq 1``
- `σ0`/`sigma0` is the initial step size `σ`
- `weights` are recombination weights, if the weights are set to ``1/\\mu`` then the *intermediate* recombination is activated.
- `metrics` is a collection of convergence metrics.
"""
struct CMAES{T} <: AbstractOptimizer
    μ::Int
    λ::Int
    c_1::T
    c_c::T
    c_μ::T
    c_σ::T
    σ₀::T
    cₘ::T
    wᵢ::Vector{T}
    metrics::ConvergenceMetrics

    function CMAES(;
            μ::Int = 10, λ::Int = 2μ, mu::Int = μ, lambda::Int = 0, weights::Vector{T} = zeros(lambda),
            c_1::Real = NaN, c_c::Real = NaN, c_mu::Real = NaN, c_sigma::Real = NaN,
            sigma0::Real = 0.5, c_m::Real = 1, metrics = ConvergenceMetric[]
        ) where {T}
        @assert c_m ≤ 1 "cₘ > 1"
        if lambda == 0
            lambda = μ == mu ? λ : 2 * mu
            weights = zeros(lambda)
        end
        if mu != lambda / 2
            mu = lambda >> 1
        end
        @assert length(weights) == lambda "Number of weights must be $lambda"
        if length(metrics) == 0
            push!(metrics, AbsDiff(T(1.0e-12)))
        end
        return new{T}(mu, lambda, c_1, c_c, c_mu, c_sigma, sigma0, c_m, weights, metrics)
    end
end
summary(m::CMAES) = "($(m.μ),$(m.λ))-CMA-ES"
show(io::IO, m::CMAES) = print(io, summary(m))
population_size(method::CMAES) = method.μ
default_options(method::CMAES) = (iterations = 1500,)

mutable struct CMAESState{T, TI} <: AbstractOptimizerState
    N::Int
    μ_eff::T
    c_1::T
    c_c::T
    c_μ::T
    c_σ::T
    fitpop::Vector{T}
    C::Matrix{T}
    s::Vector{T}
    s_σ::Vector{T}
    σ::T
    wᵢ::Vector{T}
    d_σ::T
    parent::TI
    fittest::TI
    z::Matrix{T}
end
value(s::CMAESState) = first(s.fitpop)
minimizer(s::CMAESState) = s.fittest

"""Initialization of CMA-ES algorithm state"""
function initial_state(method::CMAES, options, objfun, population)
    @unpack μ, λ, c_1, c_c, c_μ, c_σ, σ₀, cₘ, wᵢ = method
    @assert μ < λ "Offspring population must be larger then parent population"

    T = typeof(value(objfun))
    individual = first(population)
    TI = typeof(individual)
    n = length(individual)
    α_cov = 2

    # set parameters
    μ_eff = 1 / sum(w -> w^2, wᵢ[1:μ])
    if !(1 ≤ μ_eff ≤ μ)
        # default parameters for weighted recombination
        w′ᵢ = T[log((λ + 1) / 2.0) - log(i) for i in 1:λ]
        μ = sum(w′ᵢ .> 0)
        μ_eff = sum(w′ᵢ[1:μ])^2 / sum(w -> w^2, w′ᵢ[1:μ])
        μ⁻_eff = sum(w′ᵢ[(μ + 1):end])^2 / sum(w -> w^2, w′ᵢ[(μ + 1):end])
        c_1 = isnan(c_1) ? α_cov / ((n + 1.3)^2 + μ_eff) : c_1
        c_μ = isnan(c_μ) ? min(1 - c_1, α_cov * (μ_eff - 2 + 1 / μ_eff) / ((n + 2)^2 + α_cov * μ_eff / 2)) : c_μ
        c_c = isnan(c_c) ? (4 + μ_eff / n) / (n + 4 + 2 * μ_eff / n) : c_c
        c_σ = isnan(c_σ) ? (μ_eff + 2) / (n + μ_eff + 5) : c_σ
        w⁺ = sum(w′ᵢ[w′ᵢ .≥ 0])
        wᵢ = [w / w⁺ for w in w′ᵢ]
        w⁻ = sum(wᵢ[wᵢ .< 0])
        wᵢ[(μ + 1):end] ./= -w⁻
        if wᵢ[end] < 0
            if c_μ > 0
                setweights!(wᵢ, μ, 1 + c_1 / c_μ)
                boundweights!(wᵢ, μ, (1 - c_1 - c_μ) / c_μ / n)
            end
            boundweights!(wᵢ, μ, 1 + 2 * μ⁻_eff / (μ_eff + 2))
        end
    else
        c_c = isnan(c_c) ? 1 / sqrt(n) : c_c
        c_σ = isnan(c_σ) ? 1 / sqrt(n) : c_σ
        c_μ = isnan(c_μ) ? min(μ_eff / n^2, 1 - c_1) : c_μ
        c_1 = isnan(c_1) ? min(2 / n^2, 1 - c_μ) : c_1
    end

    @assert 1 ≤ μ_eff ≤ μ "μ_eff ∉ [1, μ]"
    @assert c_1 + c_μ ≤ 1 "c_1 + c_μ > 1"
    @assert c_σ < 1 "c_σ ≥ 1"
    @assert c_c ≤ 1 "c_c > 1"
    @assert 0 < wᵢ[1] ≤ 1 "0 < wᵢ[1] ≤ 1"
    @assert wᵢ[end] ≤ 0 "wᵢ[end] ≤ 0"
    @assert all(wᵢ[i] >= wᵢ[i + 1] for i in 1:(length(wᵢ) - 1)) "wᵢ are not monotonous"
    @assert wᵢ[μ] > 0 >= wᵢ[μ + 1] "There are $μ positive weights"

    d_σ = 1 + 2 * max(0, sqrt((μ_eff - 1) / (n + 1)) - 1) + c_σ

    # setup initial state
    return CMAESState{T, TI}(
        n, μ_eff, c_1, c_c, c_μ, c_σ,
        fill(convert(T, Inf), μ),
        diagm(0 => ones(T, n)),
        zeros(T, n), zeros(T, n), method.σ₀, wᵢ, d_σ,
        copy(individual), copy(individual), zeros(T, n, λ)
    )
end

function update_state!(
        objfun, constraints, state::CMAESState{T, TI}, population::AbstractVector{TI},
        method::CMAES, options, itr
    ) where {T, TI}
    μ, λ, cₘ = method.μ, method.λ, method.cₘ
    N, μ_eff, σ, w, d_σ = state.N, state.μ_eff, state.σ, state.wᵢ, state.d_σ
    c_1, c_c, c_μ, c_σ = state.c_1, state.c_c, state.c_μ, state.c_σ
    weights = view(w, 1:μ)
    evaltype = options.parallelization

    parent = reshape(state.parent, N)
    parentshape = size(state.parent)

    randn!(options.rng, state.z)
    # y = zeros(T, N, λ)
    ȳ = zeros(T, N)
    offspring = Array{Vector{T}}(undef, λ)
    fitoff = fill(Inf, λ)

    B, D = try
        F = eigen(Symmetric(state.C))
        F.vectors, Diagonal(sqrt.(max.(0, F.values)))
    catch ex
        @error "Break on eigendecomposition: $ex: $(state.C)"
        return true
    end

    C = σ * B * D
    for i in 1:λ
        # offspring are generated by transforming standard normally distributed random vectors using a transformation matrix
        offspring[i] = apply!(constraints, parent + C * state.z[:, i])
    end
    # calculate fitness of the population
    value!(objfun, fitoff, reshape.(offspring, parentshape...))
    # apply penalty to fitness
    penalty!(fitoff, constraints, offspring)

    # Select new parent population
    idx = sortperm(fitoff)
    for i in 1:μ
        o = offspring[idx[i]]
        population[i] = reshape(o, parentshape...)
        ȳ .+= (o .- parent) .* (w[i] / σ)
        state.fitpop[i] = fitoff[idx[i]]
    end

    # recombination
    z_λ = view(state.z, :, idx)
    z̄ = vec(sum((@view z_λ[:, 1:μ]) .* weights', dims = 2))
    parent += (cₘ * σ) .* ȳ  #  forming recombinant parent for next generation
    # update evolution paths
    state.s_σ .= (1 - c_σ) .* state.s_σ .+ sqrt(μ_eff * c_σ * (2 - c_σ)) * (B * z̄)
    h_σ = sum(abs2, state.s_σ) / N / (1 - (1 - c_σ)^(2 * itr)) < (2 + 4 / (N + 1))
    state.s = (1 - c_c) * state.s + (h_σ * sqrt(μ_eff * c_c * (2 - c_c))) * ȳ
    # perform rank-one update
    rank_1 = c_1 .* state.s * state.s'
    # perform rank-μ update with the steps y = B * D * z
    y_λ = B * D * z_λ
    rank_μ = c_μ * sum((w ≥ 0 ? 1 : N / norm(B * zi)^2) * w * (yi * yi') for (w, zi, yi) in zip(w, eachcol(z_λ), eachcol(y_λ)))
    # update covariance
    c1a = c_1 * (1 - (1 - h_σ) * c_c * (2 - c_c))
    state.C .= (1 - c1a - c_μ * sum(w)) .* state.C + rank_1 + rank_μ
    # adapt step-size σ
    state.σ = σ * exp(min(1, (c_σ / d_σ) * (sum(abs2, state.s_σ) / N - 1) / 2))

    state.fittest = population[1]
    state.parent = reshape(parent, parentshape...)

    return false
end

function setweights!(w, μ, val)
    @assert w[μ + 1] < 0 "There are no negative weights: $(w[(μ - 1):(μ + 1)])"
    λ = length(w)
    if w[end] >= 0
        w[(μ + 1):λ] .= -abs(val) / (λ - μ)
    end
    fct = abs(val / sum(w[(μ + 1):end]))
    w[(μ + 1):end] .*= fct
    return w
end

function boundweights!(w, μ, val)
    sum(w[(μ + 1):end]) >= -abs(val) && return w
    @assert w[end] < 0 && w[μ + 1] <= 0 "Negative weights are not set"
    fct = abs(val / sum(w[(μ + 1):end]))
    if fct < 1
        w[(μ + 1):end] .*= fct
    end
    return w
end
