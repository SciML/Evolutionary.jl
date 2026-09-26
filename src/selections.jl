# GA selections
# ==============

"""
    ranklinear(sp::Real)

Returns a rank-based fitness selection function, see [Selection Interface](@ref), with the selective pressure value `sp`.

In rank-based fitness selection, the population is sorted according to the objective values. The fitness assigned to each individual depends only on its position in the individuals rank and not on the actual objective value [^1].

Consider ``M`` the number of individuals in the population, ``P`` the position of an individual in this population (least fit individual has ``P = 1``, the fittest individual ``P = M``) and ``SP`` the selective pressure. The fitness value for an individual is calculated as:

``Fitness(P) = 2 - SP + \\frac{2(SP - 1)(P - 1)}{(M - 1)}``

Linear ranking allows values of selective pressure in [1.0, 2.0].

# References

[^1]: Baker J.E., Adaptive selection methods for genetic algorithms, In Proceedings of International Conference on Genetic Algorithms and Their Applications, pp. 100-111, 1985.
"""
function ranklinear(sp::Real)
    @assert 1.0 <= sp <= 2.0 "Selective pressure has to be in range [1.0, 2.0]."
    function rank(
            fitness::Vector{<:Real}, N::Int;
            rng::AbstractRNG = default_rng()
        )
        λ = length(fitness)
        idx = sortperm(fitness)
        ranks = zeros(λ)
        for i in 1:λ
            ranks[i] = (2 - sp + 2 * (sp - 1) * (idx[i] - 1) / (λ - 1)) / λ
        end
        return pselection(rng, ranks, N)
    end
    return rank
end

"""
    uniformranking(μ)

Returns a (μ, λ)-uniform ranking selection function, see [Selection Interface](@ref) with the best individuals parameter `μ`.

In uniform ranking, the best ``\\mu`` individuals are assigned a selection probability of ``1/\\mu`` while the rest them are discarded [^2].

# References

[^2]: Schwefel H.P., Evolution and Optimum Seeking, Wiley, New York, 1995.
"""
function uniformranking(μ::Int)
    function uniformrank(
            fitness::Vector{<:Real}, N::Int;
            rng::AbstractRNG = default_rng()
        )
        λ = length(fitness)
        idx = sortperm(fitness)
        @assert μ <= λ "μ should be no larger then $(λ)"
        ranks = zeros(length(fitness))
        for i in 1:μ
            ranks[idx[i]] = 1 / μ
        end
        return pselection(rng, ranks, N)
    end
    return uniformrank
end

"""
    roulette(fitness)

Roulette wheel (fitness proportionate, FPS) selection from `fitness` collection.

In roulette selection, the fitness level is used to associate a probability of selection with each individual. If ``f_i`` is the fitness of individual ``i`` in the population, its probability of being selected is ``p_i = \\frac{f_i}{\\Sigma_{j=1}^{M} f_j}``, where ``M`` is the number of individuals in the population.

*Note:* Best used in maximization context.

"""
function roulette(
        fitness::Vector{<:Real}, N::Int;
        rng::AbstractRNG = default_rng()
    )
    absf = abs.(fitness)
    prob = absf ./ sum(absf)
    return pselection(rng, prob, N)
end

"""
    rouletteinv(fitness)

Fitness proportionate selection (FPS) or roulette wheel for inverse `fitness` values. Best used in minimization to 0.
"""
rouletteinv(fitness::Vector{<:Real}, N::Int; kwargs...) =
    roulette(1.0 ./ fitness, N; kwargs...)

"""
    sus(fitness, N)

Stochastic universal sampling (SUS) provides zero bias and minimum spread [^3]. SUS is a development of fitness proportionate selection (FPS).
Using a comb-like ruler, SUS starts from a small random number, and chooses the next candidates from the rest of population remaining, not allowing the fittest members to saturate the candidate space. The individuals are mapped to contiguous segments of a line, such that each individual's segment is equal in size to its fitness exactly as in roulette-wheel selection. Here equally spaced pointers are placed over the line as many as there are individuals to be selected.

Consider ``N`` the number of individuals to be selected, then the distance between the pointers are ``1/N`` and the position of the first pointer is given by a randomly generated number in the range ``[0, 1/N]``.

*Note*: Best used in maximization context.

# References

[^3]: Baker, J. E., Reducing Bias and Inefficiency in the Selection Algorithm. In [ICGA2], pp. 14-21, 1987.
"""
function sus(fitness::Vector{<:Real}, N::Int; rng::AbstractRNG = default_rng())
    F = sum(abs, fitness)
    P = F / N
    start = P * rand(rng)
    pointers = [start + P * i for i in 0:(N - 1)]
    selected = Array{Int}(undef, N)
    i = c = 1
    for P in pointers
        while sum(abs, fitness[1:i]) < P
            i += 1
        end
        selected[c] = i
        c += 1
    end
    return selected
end

"""
    susinv(fitness)

Inverse fitness SUS. Best used in minimization to 0.
"""
susinv(fitness::Vector{<:Real}, N::Int; kwargs...) =
    sus(1.0 ./ fitness, N; kwargs...)

"""
    truncation(fitness, N)

Truncation selection returns first `N` of best `fitness` individuals
"""
function truncation(fitness::Vector{<:Real}, N::Int; kwargs...)
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    idx = sortperm(fitness)
    return idx[1:N]
end

"""
    tournament(groupSize; select=argmin) → f(fitness, N)

Create a [tournament-selection](https://en.wikipedia.org/wiki/Tournament_selection) function `f`
for a given `groupSize`. `f` chooses the winner (as chosen by `select`) among `N` random groups.
"""
function tournament(groupSize::Int; select = argmin)
    @assert groupSize > 0 "Group size must be positive"
    function tournamentN(
            fitness::AbstractVecOrMat{<:Real}, N::Int;
            rng::AbstractRNG = default_rng()
        )
        selection = fill(0, N)
        sFitness = size(fitness)
        d, nFitness = length(sFitness) == 1 ? (1, sFitness[1]) : sFitness
        tour = randperm(rng, nFitness)
        j = 1
        for i in 1:N
            idxs = tour[j:(j + groupSize - 1)]
            selected = d == 1 ? view(fitness, idxs) : view(fitness, :, idxs)
            winner = select(selected)
            selection[i] = idxs[winner]
            j += groupSize
            if (j + groupSize) >= nFitness && i < N
                tour = randperm(rng, nFitness)
                j = 1
            end
        end
        return selection
    end
    return tournamentN
end

"""
    random(fitness, N)

Returns a collection on size `N` of uniformly selected individuals from the population.
"""
random(fitness::Vector{<:Real}, N::Int; rng::AbstractRNG = default_rng()) =
    rand(rng, 1:length(fitness), N)

"""
    permutation(fitness, N)

Returns a permutation on size `N` of the individuals from the population.
"""
function permutation(
        fitness::Vector{<:Real}, N::Int;
        rng::AbstractRNG = default_rng()
    )
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    return randperm(rng, λ)[1:N]
end

"""
    randomoffset(fitness, N)

Returns a cycle selection on size `N` from an arbitrary position.
"""
function randomoffset(
        fitness::Vector{<:Real}, N::Int;
        rng::AbstractRNG = default_rng()
    )
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    rg = rand(rng, 1:λ)
    return [(i + rg) % λ + 1 for i in 1:N]
end

"""
    best(fitness, N)

Returns a collection of best individuals of size `N`.
"""
best(fitness::Vector{<:Real}, N::Int; kwargs...) = fill(last(findmin(fitness)), N)


# Utilities: selection
function pselection(rng::AbstractRNG, prob::Vector{<:Real}, N::Int)
    cp = cumsum(prob)
    selected = Array{Int}(undef, N)
    for i in 1:N
        j = 1
        r = rand(rng)
        while cp[j] < r
            j += 1
        end
        selected[i] = j
    end
    return selected
end
pselection(prob, N::Int) = pselection(default_rng(), prob, N)

function randexcl(rng::AbstractRNG, itr, exclude, dims::Int)
    idxs = Int[]
    while length(idxs) < dims
        j = rand(rng, itr)
        (j ∈ exclude || j ∈ idxs) && continue
        push!(idxs, j)
    end
    return idxs
end
randexcl(itr, exclude, dims::Int) = randexcl(default_rng(), itr, exclude, dims)

function twowaycomp(rc::AbstractMatrix)
    ra, ca, rb, cb = rc
    if ra < rb
        return 1
    elseif ra > rb
        return 2
    elseif ca < cb
        return 1
    else
        return 2
    end
end
