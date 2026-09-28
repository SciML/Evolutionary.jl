using Evolutionary
using Test
using Random
using StableRNGs

@testset "Multi-objective EA" begin

    rng = StableRNG(42)
    opts = Evolutionary.Options(rng = rng, iterations = 500)

    # domination
    P = [[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    @test Evolutionary.dominate(P[1], P[2]) == 1
    @test Evolutionary.dominate(P[1], P[1]) == 0
    @test Evolutionary.dominate(P[1], P[3]) == -1
    @test Evolutionary.dominate(P[3], P[4]) == 0
    @test Evolutionary.dominations(P)[:, 1] == [0, -1, 1, 1]

    # convergence
    R = reshape([10, 0, 6, 1, 2, 2, 1, 6, 0, 10], 2, 5)
    A = reshape([4, 2, 3, 3, 2, 4], 2, 3)
    B = reshape([8, 2, 4, 4, 2, 8], 2, 3)
    @test Evolutionary.igd(A, R) ≈ 3.707092031609239
    @test Evolutionary.igd(B, R) ≈ 2.59148346584763
    @test Evolutionary.spread(R) ≈ 0.0
    @test Evolutionary.spread(R[1:1, 1:4]) ≈ 3.75

    # Schaffer F2

    schafferf2(x::AbstractVector) = [x[1]^2, (x[1] - 2)^2]
    Random.seed!(rng, 1)
    result = Evolutionary.optimize(schafferf2, () -> 100randn(rng, 1), NSGA2(), opts)
    println("NSGA2:2RLT:SBX:PLM => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test isnan(Evolutionary.minimum(result))
    mvs = vcat(Evolutionary.minimizer(result)...)
    @test sum(0 .<= mvs .<= 2) / length(mvs) >= 0.8 # 80% in PO ∈ [0,2]
    #println(result)
    #println(extrema(mvs))

    function schafferf2!(F, x::AbstractVector) # in-place update
        F[1] = x[1]^2
        F[2] = (x[1] - 2)^2
        F
    end
    Random.seed!(rng, 42)
    result = Evolutionary.optimize(schafferf2!, zeros(2), () -> 10randn(rng, 1), NSGA2(), opts)
    println("NSGA2:2RLT:SBX:PLM => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test isnan(Evolutionary.minimum(result))
    mvs = vcat(Evolutionary.minimizer(result)...)
    @test sum(0 .<= mvs .<= 2) / length(mvs) >= 0.8 # 80% in PO ∈ [0,2]

    # Regression tests for NSGA-II survivor truncation (issue #132)
    # Deb et al. 2002 crowded comparison: among equal-rank individuals, LARGER
    # crowding distance wins; the truncated front's sorted positions must be
    # mapped back to population indices before selecting survivors.
    @testset "NSGA-II truncation picks crowded-comparison survivors" begin
        # crowding_distance!: boundary members of a single front get typemax
        Fm = Float64[
            0.0 0.2 0.49 0.51 0.8 1.0;
            1.0 0.8 0.51 0.49 0.2 0.0
        ]
        nf = size(Fm, 2)
        rks = zeros(Int, nf)
        cdist = zeros(Float64, nf)
        frs = Evolutionary.nondominatedsort!(rks, Fm)
        @test length(frs) == 1 && length(frs[1]) == nf
        Evolutionary.crowding_distance!(cdist, Fm, frs)
        @test count(isequal(typemax(Float64)), cdist[frs[1]]) == 2

        # White-box: run one deterministic update_state! step (identity
        # operators, fixed selection) and compare the survivors written into
        # `parents` against an independently recomputed crowded-comparison
        # oracle over the recorded combined fitness. Parents are chosen so the
        # first front (6 members after duplication) exceeds populationSize (5),
        # forcing the truncation branch, with front indices that differ from
        # their positions inside the front.
        f2(x::AbstractVector) = [x[1]^2, (x[1] - 2)^2]
        sel_first(fit, N; kwargs...) = collect(1:N)
        rng2 = StableRNG(123)
        opts2 = Evolutionary.Options(rng = rng2)
        method = NSGA2(
            populationSize = 5, crossoverRate = 0.0,
            mutationRate = 0.0, selection = sel_first
        )
        parents2 = [[0.5], [1.0], [1.5], [-0.5], [-1.0]]
        objfun = Evolutionary.EvolutionaryObjective(f2, first(parents2))
        state = Evolutionary.initial_state(method, opts2, objfun, parents2)
        Evolutionary.update_state!(
            objfun, Evolutionary.NoConstraints(), state,
            parents2, method, opts2, 1
        )

        # oracle: fronts + crowding recomputed fresh from the combined fitness
        n2 = 2 * method.populationSize
        rks2 = zeros(Int, n2)
        cd2 = zeros(Float64, n2)
        F2 = Evolutionary.nondominatedsort!(rks2, state.fitpop)
        @test length(F2[1]) > method.populationSize  # truncation branch exercised
        Evolutionary.crowding_distance!(cd2, state.fitpop, F2)
        expected = Int[]
        for fr in F2
            if length(expected) + length(fr) > method.populationSize
                order = sortperm(view(cd2, fr), rev = true)
                append!(expected, fr[order[1:(method.populationSize - length(expected))]])
                break
            else
                append!(expected, fr)
            end
        end
        @test length(expected) == method.populationSize

        # survivors (written into parents2) must match the oracle selection,
        # compared as fitness multisets
        survivors_fit = sort([f2(p) for p in parents2])
        expected_fit = sort([state.fitpop[:, i] for i in expected])
        @test survivors_fit == expected_fit
    end

    # Regression: update_state! must reorder fitpop/ranks/crowding with the
    # survivors written into parents (issue #174). Deb et al. 2002: survivors
    # carry their own objectives, rank and crowding into the next generation.
    @testset "NSGA-II keeps fitpop/ranks/crowding aligned with parents (issue #174)" begin
        f2(x::AbstractVector) = [x[1]^2, (x[1] - 2)^2]
        sel_first(fit, N; kwargs...) = collect(1:N)
        rng3 = StableRNG(123)
        opts3 = Evolutionary.Options(rng = rng3)
        method = NSGA2(
            populationSize = 5, crossoverRate = 0.0,
            mutationRate = 0.0, selection = sel_first
        )
        parents3 = [[0.5], [1.0], [1.5], [-0.5], [-1.0]]
        N = method.populationSize
        objfun = Evolutionary.EvolutionaryObjective(f2, first(parents3))
        state = Evolutionary.initial_state(method, opts3, objfun, parents3)

        # Identity operators + sel_first => offspring copy the current parents,
        # so the combined fitness before survivor reordering is [F_p F_p].
        parents_before = [copy(p) for p in parents3]
        Fp = reduce(hcat, f2.(parents_before))
        combined = hcat(Fp, Fp)
        rks = zeros(Int, 2N)
        cd = zeros(Float64, 2N)
        fronts = Evolutionary.nondominatedsort!(rks, combined)
        Evolutionary.crowding_distance!(cd, combined, fronts)
        @test length(fronts[1]) > N  # truncation branch
        fitidx = Int[]
        for fr in fronts
            if length(fitidx) + length(fr) > N
                order = sortperm(view(cd, fr), rev = true)
                append!(fitidx, fr[order[1:(N - length(fitidx))]])
                break
            else
                append!(fitidx, fr)
            end
        end
        @test length(fitidx) == N
        # Combined population genomes under identity ops: parents then parents
        allgen = vcat(parents_before, parents_before)
        expected_parents = allgen[fitidx]
        expected_fit = combined[:, fitidx]
        expected_ranks = rks[fitidx]
        expected_crowding = cd[fitidx]

        Evolutionary.update_state!(
            objfun, Evolutionary.NoConstraints(), state,
            parents3, method, opts3, 1
        )

        @test parents3 == expected_parents
        @test state.fitpop[:, 1:N] == expected_fit
        @test state.ranks[1:N] == expected_ranks
        @test state.crowding[1:N] == expected_crowding
        for i in 1:N
            @test state.fitpop[:, i] == f2(parents3[i])
        end
        for (ind, fitcol) in zip(state.fittest, eachcol(state.fitness))
            @test Vector(fitcol) == f2(ind)
        end

        # Second generation: without realignment, only offspring columns are
        # re-evaluated and parent-slot objectives stay stale.
        Evolutionary.update_state!(
            objfun, Evolutionary.NoConstraints(), state,
            parents3, method, opts3, 2
        )
        for i in 1:N
            @test state.fitpop[:, i] == f2(parents3[i])
        end
        for (ind, fitcol) in zip(state.fittest, eachcol(state.fitness))
            @test Vector(fitcol) == f2(ind)
        end
    end

    # Offspring that aren't crossed are copies: mutating them leaves the parents,
    # and the objective values stored for them, unchanged
    @testset "NSGA-II mutates copies of the parents" begin
        f3(x) = [Float64(count(x)), Float64(count(!, x))]
        method = NSGA2(populationSize = 10, crossoverRate = 0.0, mutationRate = 1.0, mutation = flip)
        opts4 = Evolutionary.Options(rng = StableRNG(1))
        parents4 = [bitrand(StableRNG(k), 20) for k in 1:10]
        objfun = Evolutionary.EvolutionaryObjective(f3, first(parents4), zeros(2))
        state = Evolutionary.initial_state(method, opts4, objfun, parents4)
        for itr in 1:5
            Evolutionary.update_state!(objfun, Evolutionary.NoConstraints(), state, parents4, method, opts4, itr)
        end
        for i in 1:10
            @test state.fitpop[:, i] == f3(parents4[i])
        end
    end

end
