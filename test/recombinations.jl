using Evolutionary
using Test
using Random
using StableRNGs

@testset "Recombinations" begin

    rng = StableRNG(42)

    compare(iter1, iter2) = length(setdiff(Set(iter1), Set(iter2))) == 0

    pop = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]

    @testset "ES" begin
        @test average(pop) ≈ [1 / 3.0, 1 / 3.0]
        @test compare((marriage(pop, rng = rng) for i in 1:100), ([p1, p2] for p1 in [0, 1.0], p2 in [0, 1.0]))
    end

    @testset "GA" begin

        Random.seed!(rng, 4)
        @test SPX([0:4;], [5:9;], rng = rng) == ([0, 1, 7, 8, 9], [5, 6, 2, 3, 4])
        Random.seed!(rng, 1)
        @test TPX([0:4;], [5:9;], rng = rng) == ([0, 1, 7, 8, 4], [5, 6, 2, 3, 9])
        Random.seed!(rng, 3)
        @test UX([0:4;], [5:9;], rng = rng) == ([0, 6, 7, 8, 9], [5, 1, 2, 3, 4])
        Random.seed!(rng, 2)
        @test SHFX([0:4;], [5:9;], rng = rng) == ([5, 6, 2, 8, 9], [0, 1, 7, 3, 4])

        Random.seed!(rng, 3)
        @test DC([0:4;], [5:9;], rng = rng) == ([5, 1, 2, 3, 9], [5, 6, 7, 3, 9])
        @test AX(pop[1], pop[2]) == ([0.5, 0.5], [0.5, 0.5])
        @test WAX([1.0, 0.5])(pop[1], pop[2]) == ([1.0, 2.0], [1.0, 2.0])

        Random.seed!(rng, 1)
        @test map(sum, HX(pop[1], pop[2])) |> collect ≈ [1.0, 1.0]
        Random.seed!(rng, 1)
        @test map(abs ∘ first ∘ diff, LX()(pop[1], pop[2])) |> collect ≈ [1.0, 1.0]
        Random.seed!(rng, 1)
        @test mapslices(diff, hcat(MILX(1.0, 0.0, 1.0)(Real[0, 0.0], Real[1, 0.0], rng = rng)...), dims = 2) |> vec ≈ [1.0, 0.0]
        Random.seed!(rng, 1)
        @test sum(sum.(SBX(0.5)(pop[1], pop[2], rng = rng) .- ([0.5, 0.7684], [0.5, 0.2316]))) ≈ 0 atol = 1.0e-10
        @test SBX(0.0)(pop[1], pop[2], rng = rng) == ([0.5, 0.5], [0.5, 0.5])

        Random.seed!(rng, 2)
        @test PMX([0:4;], [3, 2, 1, 0, 4], rng = rng) == ([3, 2, 1, 0, 4], [0, 1, 2, 3, 4])
        @test CX([1:8;], [2, 4, 6, 8, 7, 5, 3, 1]) == ([1, 2, 6, 4, 7, 5, 3, 8], [2, 4, 3, 8, 5, 6, 7, 1])
        @test CX([8, 4, 7, 3, 6, 2, 5, 1, 9, 0], collect(0:9)) == ([8, 1, 2, 3, 4, 5, 6, 7, 9, 0], [0, 4, 7, 3, 6, 2, 5, 1, 8, 9])
        Random.seed!(rng, 18)
        @test OX1([1:8;], [2, 4, 6, 8, 7, 5, 3, 1], rng = rng) == ([3, 4, 6, 8, 7, 5, 1, 2], [8, 7, 3, 4, 5, 6, 1, 2])
        Random.seed!(rng, 18)
        @test OX2([1:8;], [2, 4, 6, 8, 7, 5, 3, 1], rng = rng) == ([1, 2, 3, 4, 6, 5, 7, 8], [2, 4, 3, 8, 7, 5, 6, 1])
        Random.seed!(rng, 18)
        @test POS([1:8;], [2, 4, 6, 8, 7, 5, 3, 1], rng = rng) == ([1, 4, 6, 2, 3, 5, 7, 8], [4, 2, 3, 8, 7, 6, 5, 1])

        xo = BSX(3)
        @testset "BSX" for i in 1:100
            off1, off2 = xo(Bool[1, 0, 1, 0, 1, 0], Bool[1, 1, 1, 1, 0, 0])
            @test off1[end] == 0 && off2[end] == 0
            @test sum(off1) == 3 && sum(off2) == 3
        end

        Random.seed!(rng, 1)
        off1, off2 = SSX([0:4;], [5:9;], rng = rng)
        @test all(i -> i in 0:9, off1)
        @test all(i -> i in 0:9, off2)

    end

    @testset "DE" begin
        v1 = collect(1:10)
        v2 = fill(0, 10)

        xvr = BINX(0.0)
        @test count(!iszero, first(xvr(v1, v2))) == 1
        xvr = BINX(1.0)
        @test first(xvr(v1, v2)) == v1
        xvr = BINX(0.5)
        m1, m2 = xvr(v1, v2)
        @test m1 .+ m2 == v1

        xvr = EXPX(0.0)
        @test sum(first(xvr(v1, v2))) <= 10
        xvr = EXPX(1.0)
        @test first(xvr(v1, v2)) == v1
        xvr = EXPX(0.05)
        @test sum(first(xvr(v1, v2))) <= 19
    end

end
