using Evolutionary
using Test
using Random
using StableRNGs

@testset "Genetic Programming" begin
    rng = StableRNG(42)

    pop = 10
    terms = Terminal[:x, :y, rand]
    funcs = Function[+, -, *, /]

    t = TreeGP(pop, terms, funcs, maxdepth = 2)
    @test Evolutionary.population_size(t) == pop
    @test sort(Evolutionary.terminals(t)) == [:x, :y]
    @testset for (term, dim) in t.terminals
        @test dim == 1
    end
    @testset for (func, arity) in t.functions
        @test arity == 2
    end
    show(IOBuffer(), t)

    # population initialization
    popexp = Evolutionary.initial_population(t, rng = rng)
    @test length(popexp) == pop
    popexp = Evolutionary.initial_population(t, :(x + 1))
    @test popexp[1] == :(x + 1)

    # recursive helper functions
    h = 4
    gtr = TreeGP(pop, terms, funcs, maxdepth = 2, initialization = :grow)
    Random.seed!(rng, 8237463746)
    tmp = rand(rng, gtr, h)
    Random.seed!(rng, 8237463746)
    gt = rand(rng, gtr, h)
    @test tmp == gt
    @test Evolutionary.nodes(gt) < 2^(h + 1) - 1
    @test Evolutionary.height(gt) <= h
    ft = rand(rng, TreeGP(pop, terms, funcs, maxdepth = 2, initialization = :full), h)
    @test Evolutionary.nodes(ft) == 2^(h + 1) - 1
    @test Evolutionary.height(ft) == h
    @test all(
        i -> Evolutionary.subexpression(ft, i) !== nothing,
        0:(Evolutionary.subexpression_count(ft) - 1),
    )
    @test_throws BoundsError Evolutionary.subexpression(ft, Evolutionary.subexpression_count(ft))
    @test Evolutionary.depth(ft, :x) == 4
    Evolutionary.replace_subexpression!(ft, :z, 3)
    @test Evolutionary.subexpression(ft, 3) == :z
    @test Evolutionary.depth(ft, ft) == 0
    @test Evolutionary.depth(ft, Evolutionary.subexpression(ft, 3)) > 0
    @test Evolutionary.depth(ft, :w) == -1
    @test Evolutionary.evaluate(:y, Dict(:y => 1, :z => 2), 1.0, 2.0) == 1.0
    Evolutionary.copy_expression!(ft, gt)
    @test ft == gt
    @test Evolutionary.symbols(ft) |> sort == [:x, :y]

    # simplification
    using Evolutionary: simplify!
    @test Expr(:call, -, :x, :x) |> simplify! == 0
    @test Expr(:call, /, :x, :x) |> simplify! == 1
    @test Expr(:call, *, 0, :x) |> simplify! == 0
    @test Expr(:call, *, :x, 0) |> simplify! == 0
    @test Expr(:call, /, 0, :x) |> simplify! == 0
    @test Expr(:call, /, 0, 1) |> simplify! == 0
    @test Expr(:call, +, 0, :x) |> simplify! == :x
    @test Expr(:call, +, :x, 0) |> simplify! == :x
    @test Expr(:call, -, :x, 0) |> simplify! == :x
    @test Expr(:call, +, :x, :x) |> simplify! == Expr(:call, *, 2, :x)
    @test Expr(:call, -, 2, 1) |> simplify! == 1
    @test Expr(:call, exp, Expr(:call, log, 1)) |> simplify! == 1
    @test Expr(:call, log, Expr(:call, exp, 1)) |> simplify! == 1
    @test Expr(:call, -, Expr(:call, +, :x, 1), 2) |> simplify! == Expr(:call, +, :x, -1)
    @test Expr(:call, -, Expr(:call, +, 1, :x), 2) |> simplify! == Expr(:call, +, :x, -1)
    @test Expr(:call, +, Expr(:call, +, :x, 1), 2) |> simplify! == Expr(:call, +, :x, 3)
    @test Expr(:call, +, Expr(:call, +, 1, :x), 2) |> simplify! == Expr(:call, +, :x, 3)
    @test Expr(:call, +, Expr(:call, -, 1, :x), 2) |> simplify! == Expr(:call, -, 3, :x)
    @test Expr(:call, -, Expr(:call, -, 1, :x), 2) |> simplify! == Expr(:call, -, -1, :x)
    @test Expr(:call, +, Expr(:call, -, :x, 1), 2) |> simplify! == Expr(:call, +, :x, 1)
    @test Expr(:call, -, Expr(:call, -, :x, 1), 2) |> simplify! == Expr(:call, -, :x, 3)
    @test Expr(:call, +, :x, Expr(:call, -, 1, :x)) |> simplify! == 1
    @test Expr(:call, +, Expr(:call, -, 2, :x), :x) |> simplify! == 2
    @test Expr(:call, -, :x, Expr(:call, +, :x, :y)) |> simplify! == Expr(:call, -, :y)
    @test Expr(:call, -, :x, Expr(:call, -, :x, :y)) |> simplify! == :y
    @test Expr(:call, -, :x, Expr(:call, +, :y, :x)) |> simplify! == Expr(:call, -, :y)
    @test Expr(:call, -, Expr(:call, -, :x, :y), :x) |> simplify! == Expr(:call, -, :y)
    @test Expr(:call, -, Expr(:call, +, :x, :y), :x) |> simplify! == :y
    @test Expr(:call, +, 2, Expr(:call, +, 1, :x)) |> simplify! == Expr(:call, +, 3, :x)
    @test Expr(:call, +, 2, Expr(:call, -, 1, :x)) |> simplify! == Expr(:call, -, 3, :x)
    @test Expr(:call, +, 2, Expr(:call, +, :x, 1)) |> simplify! == Expr(:call, +, 3, :x)
    @test Expr(:call, +, 2, Expr(:call, -, :x, 1)) |> simplify! == Expr(:call, +, 1, :x)
    @test Expr(:call, -, 2, Expr(:call, +, 1, :x)) |> simplify! == Expr(:call, -, 1, :x)
    @test Expr(:call, -, 2, Expr(:call, +, :x, 1)) |> simplify! == Expr(:call, -, 1, :x)
    @test Expr(:call, -, 1, Expr(:call, -, 2, :x)) |> simplify! == Expr(:call, +, -1, :x)
    @test Expr(:call, -, 2, Expr(:call, -, :x, 1)) |> simplify! == Expr(:call, -, 3, :x)

    # evaluation
    ex = Expr(:call, +, 1, :x) |> Evolutionary.Expression
    xs = rand(rng, 10)
    @test ex(xs[1]) == xs[1] + 1
    @test ex.(xs) == xs .+ 1
    io = IOBuffer()
    show(io, "text/latex", ex)
    @test String(take!(io)) == "\\left(1.0+x\\right)"
    Evolutionary.infix(io, ex)
    @test String(take!(io)) == "(1.0+x)"

    # protected operations
    @test Evolutionary.pdiv(1, 0) == 1.0e7 + 1
    @test Evolutionary.aq(1, 0) == 1.0
    @test Evolutionary.pexp(40) == 1.0e16 + 40
    @test Evolutionary.plog(0) == -1.0e7
    @test Evolutionary.psin(Inf) == 1.0e7
    @test Evolutionary.pcos(Inf) == 1.0e7
    @test Evolutionary.ppow(10, 20) == 1.0e7 + 30
    @test Evolutionary.ppow(-4, 0.5) == 2.0
    @test Evolutionary.ppow(2, 2) == 4.0
    @test Evolutionary.cond(3, 1, 2) == 1
    @test Evolutionary.cond(-3, 1, 2) == 2

    # symreg
    fitfun(x) = x * x + x + 1.0
    rg = -5.0:0.1:5.0
    ys = fitfun.(rg)
    function fitobj(expr)
        ex = Evolutionary.Expression(expr)
        yy = ex.(rg)
        sum(v -> isnan(v) ? 1.0 : v, abs2.(ys .- yy)) / length(rg)
    end

    Random.seed!(rng, 6)
    res = Evolutionary.optimize(
        fitobj,
        TreeGP(
            50, Terminal[:x, randn], Function[+, -, *, Evolutionary.aq],
            mindepth = 1,
            maxdepth = 3,
            simplify = Evolutionary.simplify!,
            selection = tournament(3),
        ),
        Evolutionary.Options(show_trace = false, rng = rng, iterations = 50)
    )
    @test minimum(res) < 1.5

end
