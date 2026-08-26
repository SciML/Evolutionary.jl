using Evolutionary
using Test
using Random
using StableRNGs

@testset "Mutations" begin

    rng = StableRNG(42)

    @testset "ES Strategies" begin
        ss = [
            IsotropicStrategy{Float64}(1.0, 1, 0),
            IsotropicStrategy{Float64}(2.0, 0, 0),
            IsotropicStrategy{Float64}(3.0, 0, 0),
        ]
        @test average(ss).σ == ss[2].σ
        Random.seed!(rng, 1)
        @test gaussian(ss[1], rng = rng).σ == ss[1].σ
        @test ss[1].σ != 1.0

        ss = [
            AnisotropicStrategy{Float64}(fill(1.0, 3), 1, 1),
            AnisotropicStrategy{Float64}(fill(2.0, 3), 0, 0),
            AnisotropicStrategy{Float64}(fill(3.0, 3), 0, 0),
        ]
        @test average(ss).σ == ss[2].σ
        Random.seed!(rng, 1)
        @test gaussian(ss[1], rng = rng).σ == ss[1].σ
        @test ss[1].σ != fill(1.0, 3)
    end

    @testset "GA" begin

        Random.seed!(rng, 1)
        @testset for i in 1:100
            from, to = Evolutionary.randseg(rng, 5)
            @test from < to
        end

        Random.seed!(rng, 1)
        @test flip(falses(5), rng = rng) == [0, 0, 0, 1, 0]
        @test bitinversion(falses(5)) == ones(5)

        Random.seed!(rng, 1)
        @test all(-1 .<= uniform()(zeros(10), rng = rng) .<= 1)
        @test all(-2 .<= gaussian()(zeros(10), rng = rng) .<= 2)
        @test all(-0.5 .<= BGA(ones(10))(zeros(10), rng = rng) .<= 0.5)

        Random.seed!(rng, 1)
        lx = [0.0, 0.0]
        ux = [2.0, 1.0]
        @test PM(lx, ux, Inf)([1.0, 2.0], rng = rng)[1] ∈ [0.0, 2.0]
        @test PM(lx, ux, Inf)([1.0, 2.0], rng = rng)[2] == 1.0
        @test MIPM(lx, ux, Inf, 1.0)(Real[1.0, 2], rng = rng)[1] ∈ [0.0, 2.0]
        @test MIPM(lx, ux, 10.0, Inf)(Real[1.0, 2], rng = rng)[2] == 1.0

        Random.seed!(rng, 1)
        ex = PLM(ones(1000), pm = 1.0)(zeros(1000), rng = rng) |> extrema
        @test ex[1] >= -1
        @test ex[2] <= 1

        Random.seed!(rng, 4)
        @test inversion([0:5;], rng = rng) == [0, 1, 4, 3, 2, 5]
        Random.seed!(rng, 42)
        @test inversion(Bool[1, 0, 1, 0, 1], rng = rng) == Bool[0, 1, 1, 0, 1]
        Random.seed!(rng, 4)
        @test insertion([0:5;], rng = rng) == [0, 1, 3, 4, 2, 5]
        Random.seed!(rng, 5)
        @test swap2([0:5;], rng = rng) == [0, 1, 2, 3, 5, 4]
        Random.seed!(rng, 2)
        @test scramble([0:5;], rng = rng) == [0, 4, 1, 5, 3, 2]
        Random.seed!(rng, 2)
        @test shifting([0:5;], rng = rng) == [2, 3, 4, 5, 0, 1]
        Random.seed!(rng, 2)
        @test replacement([0:9;])([0:5;], rng = rng) == [7, 1, 2, 3, 4, 9]

    end

    @testset "DE" begin
        rec = ones(2)
        mut = [[0.5, 1.0], [1.0, 0.5]]
        @test_throws AssertionError Evolutionary.differentiation(rec, mut[1:1])
        @test Evolutionary.differentiation(rec, mut) == [0.5, 1.5]
    end

    @testset "GP" begin
        Random.seed!(rng, 2)
        H = 2
        tr = TreeGP(maxdepth = H, initialization = :grow)
        ex = rand(rng, tr, H)

        # subtree mutation does not produce offspring expression longer then the parent
        mut = subtree(tr)
        Random.seed!(rng, 1)
        off = [mut(copy(ex), rng = rng) for i in 1:10]
        @testset "Offspring Height" for i in 1:10
            map(o -> mut(o, rng = rng), off) # mutate offspring
            @test all(o -> Evolutionary.height(o) <= H, off)
        end

        # subtree mutation does not produce offspring expression longer then the parent
        mut = subtree(tr; growth = 0.3)
        Random.seed!(rng, 2)
        off = [mut(copy(ex), rng = rng) for i in 1:10]
        @testset "Offspring Height (Growth)" for i in 1:5
            map(o -> mut(o, rng = rng), off) # mutate offspring
            h = map(Evolutionary.height, off)
            @test all(h .<= 3i)
        end

        # hoist
        mut = subtree(tr; growth = 0.5)
        Random.seed!(rng, 1)
        off = [mut(copy(ex), rng = rng) for i in 1:10]
        for i in 1:5
            map(o -> mut(o, rng = rng), off)
        end
        mut = hoist(tr)
        @testset "Hoist Mutation" for i in 1:10
            n = Evolutionary.nodes(off[i])
            m = Evolutionary.nodes(mut(off[i], rng = rng))
            @test n >= m
        end

        # shrink
        mut = subtree(tr; growth = 0.6)
        Random.seed!(rng, 1)
        off = [mut(copy(ex), rng = rng) for i in 1:10]
        for i in 1:5
            map(o -> mut(o, rng = rng), off)
        end
        mut = shrink(tr)
        @testset "Shrink Mutation" for i in 1:10
            n = Evolutionary.nodes(off[i])
            m = Evolutionary.nodes(mut(off[i], rng = rng))
            @test n >= m
        end

        # point
        mut = point(tr)
        Random.seed!(rng, 1)
        @testset "Point Mutation" for i in 1:10
            mex = mut(copy(ex), rng = rng)
            @test Evolutionary.nodes(ex) == Evolutionary.nodes(mex)
            @test sum(x == y for (x, y) in zip(ex.args, mex.args)) >= 2
        end

    end
end
