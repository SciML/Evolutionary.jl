# Mutation operators
# ==================

# Evolutionary mutations
# ======================

"""
    nop(s::AbstractStrategy)

This is a dummy mutation operator that does not change the strategy.
"""
nop(recombinant::AbstractVector, s::AbstractStrategy; kwargs...) = recombinant

"""
    gaussian(x, s::IsotropicStrategy)

Performs Gaussian isotropic mutation of the recombinant `x` given the strategy `s`  by adding Gaussian noise as follows:

- ``x_i^\\prime = x_i + s.\\sigma \\mathcal{N}_i(0,1)``

"""
function gaussian(
        recombinant::AbstractVector, s::IsotropicStrategy;
        rng::AbstractRNG = default_rng()
    )
    vals = randn(rng, length(recombinant)) * s.σ
    recombinant += vals
    return recombinant
end

"""
    gaussian(x, s::AnisotropicStrategy)

Performs Gaussian anisotropic mutation of the recombinant `x` given the strategy `s`  by adding Gaussian noise as follows:

- ``x_i^\\prime = x_i + s.\\sigma_i \\mathcal{N}_i(0,1)``

"""
function gaussian(
        recombinant::AbstractVector, s::AnisotropicStrategy;
        rng::AbstractRNG = default_rng()
    )
    @assert length(s.σ) == length(recombinant) "Parameter `σ` must be defined for every dimension of objective parameter"
    vals = randn(rng, length(recombinant)) .* s.σ
    recombinant += vals
    return recombinant
end

"""
    cauchy(x, s::IsotropicStrategy)

Performs isotropic mutation of the recombinant `x` given the strategy `s`  by adding a noise from the Cauchy distribution as follows:

- ``x_i^\\prime = x_i + s.\\sigma_i \\delta_i``

where ``\\delta`` is a Cauchy random variable with the scale parameter ``t = 1`` [^2].

# References

[^2]: Yao, Xin, and Yong Liu, "Fast evolution strategies", In International Conference on Evolutionary Programming, 149-161, Springer, 1997.
"""
function cauchy(
        recombinant::AbstractVector, s::IsotropicStrategy;
        rng::AbstractRNG = default_rng()
    )
    l = length(recombinant)
    vals = s.σ * randn(rng, l) ./ (randn(rng, l) .+ eps())
    recombinant += vals
    return recombinant
end


# Strategy mutation operators
# ===========================

"""
    nop(s::AbstractStrategy)

This is a dummy operator that does not change strategy.
"""
nop(s::AbstractStrategy; kwargs...) = s


"""
    gaussian(s::IsotropicStrategy)

Performs in-place mutation of the isotropic strategy `s` modifying its mutated strategy parameter ``\\sigma`` with Gaussian noise as follows:

- ``\\sigma^\\prime = \\sigma \\exp(\\tau_0 \\mathcal{N}(0,1))``

"""
function gaussian(s::IsotropicStrategy; rng::AbstractRNG = default_rng())
    s.σ *= exp(s.τ₀ * randn(rng))
    return s
end


"""
    gaussian(s::AnisotropicStrategy)

Performs in-place mutation of the anisotropic strategy `s` modifying its mutated strategy parameter ``\\sigma`` with Gaussian noise as follows:

- ``\\sigma_i^\\prime = \\sigma_i \\exp(\\tau_0 \\mathcal{N}(0,1) + \\tau_i \\mathcal{N}(0,1))``

"""
function gaussian(s::AnisotropicStrategy; rng::AbstractRNG = default_rng())
    s.σ .*= exp.(s.τ₀ * randn(rng)) * exp.(s.τ * randn(rng, length(s.σ)))
    return s
end


# Genetic mutations
# =================

# Binary mutations
# ---------------------

"""
    genop(recombinant)

This is a dummy mutation operator that does not change the `recombinant`.
"""
genop(recombinant; kwargs...) = recombinant

"""
    flip(recombinant)

Returns an in-place mutated binary `recombinant` with a bit flips at random positions.
"""
function flip(
        recombinant::T;
        rng::AbstractRNG = default_rng()
    ) where {T <: AbstractVector{Bool}}
    s = length(recombinant)
    pos = rand(rng, 1:s)
    recombinant[pos] = !recombinant[pos]
    return recombinant
end

"""
    bitinversion(recombinant)

Returns an in-place mutated binary `recombinant` with its bits inverted.
"""
bitinversion(recombinant::T) where {T <: AbstractVector{Bool}} = map!(!, recombinant, recombinant)


# Real-valued mutations
# ---------------------

"""
    uniform(r = 1.0)

Returns an in-place real valued mutation function that performs the uniform distributed mutation [^1].

The mutation operator randomly chooses a number ``z`` in from the uniform distribution on the interval ``[-r,r]``, the mutation range.
The mutated individual is given by

- ``x_i^\\prime = x_i + z_i``

# References

[^1]: Mühlenbein, H. and Schlierkamp-Voosen, D., "Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization", Evolutionary Computation, 1 (1), 25-49, 1993.
"""
function uniform(r::Real = 1.0)
    function mutation(
            recombinant::T;
            rng::AbstractRNG = default_rng()
        ) where {T <: AbstractVector}
        d = length(recombinant)
        recombinant .+= 2r .* rand(rng, d) .- r
        return recombinant
    end
    return mutation
end

"""
    gaussian(σ = 1.0)

Returns an in-place real valued mutation function that performs the normal distributed mutation [^1].

The mutation operator randomly chooses a number ``z`` in from the normal distribution ``\\mathcal{N}(0,\\sigma)`` with standard deviation ``\\sigma``.
The mutated individual is given by

- ``x_i^\\prime = x_i + z_i``

"""
function gaussian(σ::Real = 1.0)
    function mutation(
            recombinant::T;
            rng::AbstractRNG = default_rng()
        ) where {T <: AbstractVector}
        d = length(recombinant)
        recombinant .+= σ .* randn(rng, d)
        return recombinant
    end
    return mutation
end

"""
    BGA(valrange, m = 20)

Returns an in-place real valued mutation function that performs the BGA mutation scheme with the mutation range `valrange` and the mutation probability `1/m` [^1].
"""
function BGA(valrange::Vector, m::Int = 20)
    prob = 1.0 / m
    function mutation(
            recombinant::T;
            rng::AbstractRNG = default_rng()
        ) where {T <: AbstractVector}
        d = length(recombinant)
        @assert length(valrange) == d "Range matrix must have $(d) columns"
        δ = zeros(m)
        for i in 1:length(recombinant)
            for j in 1:m
                δ[j] = (rand(rng) < prob) ? δ[j] = 2.0^(-j) : 0.0
            end
            if rand(rng, Bool)
                recombinant[i] += sum(δ) * valrange[i]
            else
                recombinant[i] -= sum(δ) * valrange[i]
            end
        end
        return recombinant
    end
    return mutation
end

"""
    PM(lower, upper, p = 2)

Returns an in-place real valued mutation function that performs the Power Mutation (PM)
scheme within `lower` and `upper` bound, and an index of
mutation `p`[^3].

*Note:* The implementation is a degenerate case of Mixed Integer Power Mutation ([`MIPM`](@ref))

# References

[^3]: K. Deep, M. Thakur, "A new crossover operator for real coded genetic algorithms", Applied Mathematics and Computation 188, 895-912, 2007.
"""
function PM(lower::Vector, upper::Vector, p::Float64 = 5.0) # index of distribution p
    return mipmmutation(lower, upper, p)
end

"""
    MIPM(lower, upper, p_real = 10, p_int = 4)

Returns an in-place real valued mutation function that performs the Mixed Integer Power Mutation (MI-PM) scheme within `lower` and `upper` bound, and an index of mutation `p_real` for real value and `p_int` for integer values[^4].

# References

[^4]: K. Deep, K. P. Singh, M. L. Kansal, and C. Mohan, "A real coded  genetic algorithm for solving integer and mixed integer optimization problems", Appl. Math. Comput. 212, 505-518, 2009
"""
function MIPM(lowerBounds::Vector, upperBounds::Vector, p_real::Float64 = 10.0, p_int::Float64 = 4.0) # index of distribution p
    return mipmmutation(lowerBounds, upperBounds, p_real, p_int)
end

function mipmmutation(
        lowerBounds::Vector, upperBounds::Vector,
        p_real::Float64, p_int::Union{Nothing, Float64} = nothing
    )
    function pm_mutation(rng, x, l, u, s, d)
        x̄ = d < rand(rng) ? x - s * (x - l) : x + s * (u - x)
        return if isa(x, Integer)
            if isinteger(x̄)
                Int(x̄)
            else
                floor(Int, x̄) + (rand(rng) > 0.5)
            end
        else
            x̄
        end
    end
    function mutation(
            recombinant::T;
            rng::AbstractRNG = default_rng()
        ) where {T <: Vector}
        d = length(recombinant)
        @assert length(lowerBounds) == d "Bounds vector must have $(d) columns"
        @assert length(upperBounds) == d "Bounds vector must have $(d) columns"
        @assert !(p_int === nothing && any(isa(x, Integer) for x in recombinant)) "Need to set p_int for integer variables"
        u = rand(rng)
        P = (isa(x, Integer) ? p_int : p_real for x in recombinant)
        S = u .^ (1 ./ P) # random var following power distribution
        D = (recombinant - lowerBounds) ./ (upperBounds - lowerBounds)
        broadcast!(
            (x, l, u, s, d) -> pm_mutation(rng, x, l, u, s, d),
            recombinant, recombinant, lowerBounds, upperBounds, S, D
        )
        return recombinant
    end
    return mutation
end

"""
    PLM(lower, upper, η = 2)

Returns an in-place real valued mutation function that performs the Polynomial Mutation (PLM) scheme
within `lower` and `upper` bounds, and a mutation distribution index `η`[^9].

# References

[^9]: K. Deb, R. B. Agrawal, "Simulated Binary Crossover for Continuous Search Space", Complex Syst., 9., 1995
"""
function PLM(Δ::Union{Real, Vector} = 1.0; η = 2, pm::Real = NaN) # index of distribution p
    function mutation(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
        d = length(recombinant)
        pm = isnan(pm) ? 1 / d : pm
        mask = rand(rng, d) .< pm
        u = rand(rng, d)
        mask_p = u .<= 0.5
        mask_n = u .> 0.5
        δpw = 1 / (η + 1)
        δ = similar(recombinant)
        δ[mask_p] .= (2 * u[mask_p]) .^ δpw .- 1
        δ[mask_n] .= 1 .- (2(1 .- u[mask_n])) .^ δpw
        recombinant[mask] .+= (Δ .* δ)[mask]
        return recombinant
    end
    return mutation
end
PLM(lower::Vector, upper::Vector; η::Real = 2, pm::Real = NaN) = PLM(upper - lower; η = η, pm = pm)


# Combinatorial mutations (applicable to binary vectors)
# ------------------------------------------------------

"""
    inversion(recombinant)

Returns an in-place mutated individual with a random arbitrary length segment of the genome in the reverse order.
"""
function inversion(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = randseg(rng, l)
    l = round(Int, (to - from) / 2)
    if from + 1 == to
        swap!(recombinant, from, to)
    else
        for i in 0:(l - 1)
            swap!(recombinant, from + i, to - i)
        end
    end
    return recombinant
end

"""
    insertion(recombinant)

Returns an in-place mutated individual with an arbitrary element of the genome moved in a random position.
"""
function insertion(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = randseg(rng, l)
    val = recombinant[from]
    deleteat!(recombinant, from)
    return insert!(recombinant, to, val)
end

"""
    swap2(recombinant)

Returns an in-place mutated individual with a two random elements of the genome are swapped.
"""
function swap2(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = randseg(rng, l)
    swap!(recombinant, from, to)
    return recombinant
end

"""
    scramble(recombinant)

Returns an in-place mutated individual with elements, on a random arbitrary length segment of the genome, been scrambled.
"""
function scramble(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = randseg(rng, l)
    diff = to - from + 1
    if diff > 1
        patch = recombinant[from:to]
        idx = randperm(rng, diff)
        for i in 1:diff
            recombinant[from + i - 1] = patch[idx[i]]
        end
    end
    return recombinant
end

"""
    shifting(recombinant)

Returns an in-place mutated individual with a random arbitrary length segment of the genome been shifted to an arbitrary position.
"""
function shifting(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
    l = length(recombinant)
    from, to, where = sort(rand(rng, 1:l, 3))
    patch = recombinant[from:to]
    diff = where - to
    if diff > 0
        # move values after tail of patch to the patch head position
        for i in 1:diff
            recombinant[from + i - 1] = recombinant[to + i]
        end
        # place patch values in order
        start = from + diff
        for i in 1:length(patch)
            recombinant[start + i - 1] = patch[i]
        end
    end
    return recombinant
end

"""
    replacement(pool; minchange = 1)(recombinant)

Replacement mutation operator changes an arbitrary number, no smaller then `minchange`,
of elements in the individual by replacing them with elements from the predefined `pool` that are not in the individual.
"""
function replacement(pool::Vector{P}; minchange = 1) where {P}
    function rplc(recombinant::T; rng::AbstractRNG = default_rng()) where {T <: AbstractVector}
        l = length(recombinant)
        p = length(pool)
        # how many values to change
        nchg = max(minchange, rand(rng, 1:min(l, p - l)))
        # select new elements
        idxs = randperm(rng, p)
        new_vals = P[]
        for i in idxs
            if pool[i] ∉ recombinant
                push!(new_vals, pool[i])
            end
            length(new_vals) == nchg && break
        end
        # update arbitrary positions with new values
        new_idxs = randperm(rng, l)[1:nchg]
        recombinant[new_idxs] .= new_vals
        return recombinant
    end
    return rplc
end


# Differential Evolution
# ======================

"""
    differentiation(recombinant, mutators; F = 1.0)

Returns an in-place differently mutated individual ``x^\\prime`` from `recombinant` ``x``  by `mutators` ``\\{\\xi_1, \\ldots, \\xi_n \\}`` as follows

- ``x^\\prime = x + \\sum_{i=1}^{n/2} F (\\xi_{2i-1} - \\xi_{2i})``

"""
function differentiation(recombinant::T, mutators::AbstractVector{T}; F::Real = 1.0) where {T <: AbstractVector}
    m = length(mutators)
    @assert m % 2 == 0 "Must be even number of target mutators"
    for i in 1:2:m
        recombinant .+= F .* (mutators[i] .- mutators[i + 1])
    end
    return recombinant
end


# Genetic Programming
# ======================

"""
    subtree(method::TreeGP; growth::Real = 0.0)

Returns an in-place expression mutation function that performs mutation of an arbitrary expression subtree with a randomly generated one [^5].

Parameters:
- `growth`: Growth restriction on the offspring in comparison to the parent.
            The offspring cannot be more than `growth`% deeper than its parent.  (default: `0.0`)

# References

[^5]: K. E. Kinnear, Jr., "Evolving a sort: Lessons in genetic programming", In Proceedings of the 1993 International Conference on Neural Networks, vol 2, 881-888, IEEE Press, 1993.
"""
function subtree(method::TreeGP; growth::Real = 0.0)
    function mutation(recombinant::Expr; rng::AbstractRNG = default_rng())
        i = rand(rng, 1:(subexpression_count(recombinant) - 1))
        th = depth(recombinant, subexpression(recombinant, i))
        maxh = if growth > 0
            rh = height(recombinant)
            mh = max(method.maxdepth, rand(rng, rh:(rh * (1 + growth))))
            round(Int, mh)
        else
            method.maxdepth
        end
        replace_subexpression!(recombinant, rand(rng, method, max(0, maxh - th)), i)
        return recombinant
    end
    return mutation
end

"""
    point(method::TreeGP)

Returns an in-place expression mutation function that replaces an arbitrary node in the tree by the randomly selected one.
Node replacement mutation is similar to bit string mutation in that it randomly changes a point in the individual.
To ensure the tree remains legal, the replacement node has the same number of arguments as the node it is replacing [^6].

# References

[^6]: B. McKay, M. J. Willis, and G. W. Barton., "Using a tree structured genetic algorithm to perform symbolic regression", GALESIA, vol 414, 487-492, 1995.
"""
function point(method::TreeGP)
    function mutation(recombinant::Expr; rng::AbstractRNG = default_rng())
        i = rand(rng, 0:(subexpression_count(recombinant) - 1))
        nd = subexpression(recombinant, i)
        if isa(nd, Expr)
            aty = length(nd.args) - 1
            atyfnc = filter(kv -> kv[2] == aty, method.functions)
            if length(atyfnc) > 0
                nd.args[1] = atyfnc |> keys |> rand
            end
        else
            replace_subexpression!(recombinant, randterm(rng, method), i)
        end
        return recombinant
    end
    return mutation
end

"""
    hoist(method::TreeGP)

Returns an in-place expression mutation function that creates a new offspring individual which is copy
of a randomly chosen subtree of the parent. Thus, the offspring will be smaller than the parent
and will have a different root node [^7].

# References

[^7]: K. E. Kinnear, Jr., "Fitness landscapes and difficulty in genetic programming", In Proceedings of the 1994 IEEE World Conference on Computational Intelligence, vol 1, 142-147, IEEE Press, 1994.
"""
function hoist(method::TreeGP)
    function mutation(recombinant::Expr; rng::AbstractRNG = default_rng())
        rnodes = subexpression_count(recombinant)
        stsize = 0
        ch = recombinant
        while stsize < 2 && rnodes > 3
            i = rand(rng, 1:(rnodes - 1))
            ch = subexpression(recombinant, i)
            stsize = subexpression_count(ch)
        end
        return copy_expression!(recombinant, ch)
    end
    return mutation
end

"""
    shrink(method::TreeGP)

Returns an in-place expression mutation function that replaces a randomly chosen subtree with a randomly
created terminal. This is a special case of subtree mutation where the replacement tree is a terminal.
As with hoist mutation, it is motivated by the desire to reduce program size [^8].

# References

[^8]: P. J. Angeline, "An investigation into the sensitivity of genetic programming to the frequency of leaf selection during subtree crossover", Genetic Programming 1996: Proceedings of the First Annual Conference, 21–29, 1996.
"""
function shrink(method::TreeGP)
    function mutation(recombinant::Expr; rng::AbstractRNG = default_rng())
        i = rand(rng, 1:(subexpression_count(recombinant) - 1))
        replace_subexpression!(recombinant, randterm(rng, method), i)
        return recombinant
    end
    return mutation
end


# Utilities
# =====
function randseg(rng::AbstractRNG, l)
    from, to = rand(rng, 1:l, 2)
    if from == to
        if to < l
            to += 1
        else
            from -= 1
        end
    elseif from > to
        from, to = to, from
    end
    return (from, to)
end

function swap!(v::T, from::Int, to::Int) where {T <: AbstractVector}
    val = v[from]
    v[from] = v[to]
    return v[to] = val
end

"""
    mutationwrapper(gamutation)

Wrap a genetic-algorithm mutation so it can be used as an evolution-strategy mutation.
"""
function mutationwrapper(gamutation::Function)
    wrapper(recombinant::T, s::S; kwargs...) where {T <: AbstractVector, S <: AbstractStrategy} = gamutation(recombinant; kwargs...)
    return wrapper
end
