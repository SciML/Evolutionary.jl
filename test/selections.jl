using Evolutionary
using Test
using Random
using Statistics
using StableRNGs

@testset "Selections" begin

    rng = StableRNG(42)

    @testset "Rank" begin
        fitness = [5, 1, 3, 4, 2.0]

        s = ranklinear(1.0)
        Random.seed!(rng, 2)
        @test s(fitness, 2, rng = rng) == [5, 1]

        s = ranklinear(2.0)
        Random.seed!(rng, 2)
        @test s([1, 2], 2, rng = rng) == [2, 2]
    end

    @testset "Uniform" begin
        s = uniformranking(2)
        Random.seed!(rng, 2)
        @test sort(unique(s([1.0, 2.0, 3.0], 10, rng = rng))) == [1, 2]
        Random.seed!(rng, 2)
        @test sort(unique(s([5, 2, 3], 5, rng = rng))) == [2, 3]
        @test_throws AssertionError s([1.0], 4)
    end

    @testset "Roulette" begin
        Random.seed!(rng, 2)
        @test roulette([30.0, -0.1, -0.2, -30.0], 2, rng = rng) == [4, 1]
        Random.seed!(rng, 2)
        @test roulette([0.0, 0.0, 3.0], 2, rng = rng) == [3, 3]
        Random.seed!(rng, 2)
        @test roulette([0, 2, 0], 2, rng = rng) == [2, 2]
        Random.seed!(rng, 2)
        @test rouletteinv([0.0, 0.0, 3.0], 2, rng = rng) == [1, 1]
        Random.seed!(rng, 2)
        @test rouletteinv([1, 2, 0], 2, rng = rng) == [3, 3]
    end

    @testset "Truncation" begin
        @test truncation([0.0, 2.0, 3.0], 2) == [1, 2]
        @test truncation([0, 2, 1], 1) == [1]
        @test_throws AssertionError truncation([1.0, 2.0], 5)
    end

    @testset "Tournament" begin
        @test_throws AssertionError tournament(0)
        t = tournament(3, select = argmax)
        Random.seed!(rng, 2)
        @test all(t([0, 2, 0], 100, rng = rng) .== 2)
        t = tournament(3)
        Random.seed!(rng, 2)
        @test all(i -> i ∈ [1, 3], t([0, 2, 0], 100, rng = rng))
        t = tournament(2)
        Random.seed!(rng, 2)
        @test all(t([0.0, 0.0, 1.0], 100, rng = rng) .< 3.0)
        fitness = [0 0 1 1; 0 1 0 1]
        t = tournament(2, select = Evolutionary.twowaycomp)
        Random.seed!(rng, 2)
        @test all(t(fitness, 100, rng = rng) .< 4)
        @test mean(t(fitness, 100, rng = rng)) < 2
    end

    @testset "SUS" begin
        Random.seed!(rng, 2)
        @test sort(unique(sus([1.0, 0.0, 2.0], 5, rng = rng))) == [1, 3]
        Random.seed!(rng, 2)
        @test sort(unique(sus([0, 1, 2], 5, rng = rng))) == [2, 3]
    end

    @testset "DE" begin
        N = 5
        col = collect(1:10)

        Random.seed!(rng, 1)
        idxs = random(col, N, rng = rng)
        @test length(idxs) == N
        @test all(i -> i ∈ col, idxs)
        @test !all(length(unique(random(col, N, rng = rng))) == N for i in 1:100)

        Random.seed!(rng, 1)
        idxs = randomoffset(col, N, rng = rng)
        @test length(idxs) == N
        @test all(i -> i ∈ col, idxs)
        @test all(
            let idxs = randomoffset(col, N, rng = rng)
                let l = (idxs[1] - 1 + N) % 10
                    idxs[N] == (l == 0 ? 10 : l)
                end
            end for i in 1:100
        )

        idxs = best(reverse(col), N)
        @test length(idxs) == N
        @test all(i -> i ∈ col, idxs)
        @test idxs == fill(10, N)

        Random.seed!(rng, 1)
        idxs = permutation(col, N, rng = rng)
        @test length(idxs) == N
        @test all(i -> i ∈ col, idxs)
        @test all(length(unique(permutation(col, N, rng = rng))) == N for i in 1:100)

        Random.seed!(rng, 1)
        @testset "Mutually Exclusive Indices" for i in idxs
            targets = Evolutionary.randexcl(rng, col, [i], 4)
            @test length(unique(push!(targets, i))) == N
        end
    end

end
