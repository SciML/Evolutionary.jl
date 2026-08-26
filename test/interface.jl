using Evolutionary
using NLSolversBase: f_calls, value, value!, value!!
using Test
include("types.jl")

@testset "API" begin

    #########
    # SETUP #
    #########

    using Evolutionary: BoxConstraints, PenaltyConstraints,
        EvolutionaryObjective

    # objective function
    dimension = 5
    individual = ones(dimension)
    minval = 1.0
    func = x -> sum(x) + 1
    objfun = EvolutionaryObjective(func, individual)
    st = TestOptimizerState(individual, minval)
    mthd = TestOptimizer()
    cnstr = Evolutionary.NoConstraints()

    # options
    opts = Evolutionary.Options(store_trace = true; default_options(mthd)...)
    show(IOBuffer(), opts)

    # population
    ppl_size = 7
    ppl = [rand(dimension) for i in 1:ppl_size]

    #########
    # TRACE #
    #########
    @test value(st) == minval
    val = value!(objfun, rand(dimension))

    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test !Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts)
    @test tr[1].iteration == 1
    @test tr[1].value == value(st)
    @test tr[1].metadata["time"] <= time()
    @test !update_state!(objfun, cnstr, st, ppl, mthd, opts, 10)
    @test !Evolutionary.trace!(tr, 2, objfun, st, ppl, mthd, opts)
    @test tr[2].iteration == 2
    @test tr[2].value >= 1
    @test tr[2].metadata["time"] >= tr[1].metadata["time"]
    show(IOBuffer(), tr)

    # callback on state
    opts = Evolutionary.Options(store_trace = false, callback = (s -> s.iteration == 2))
    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts) == false
    @test Evolutionary.trace!(tr, 2, objfun, st, ppl, mthd, opts) == true

    # callback on trace
    opts = Evolutionary.Options(store_trace = true, callback = (tr -> length(tr) > 1))
    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts) == false
    @test Evolutionary.trace!(tr, 2, objfun, st, ppl, mthd, opts) == true

    # custom trace
    function Evolutionary.trace!(record::Dict{String, Any}, objfun, state::TestOptimizerState, population, method, options)
        record["ppl_size"] = length(population)
    end
    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test !Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts)
    @test tr[1].iteration == 1
    @test tr[1].metadata["ppl_size"] == length(ppl)


    ##########
    # RESULT #
    ##########
    opts = Evolutionary.Options(store_trace = true)
    res = Evolutionary.EvolutionaryOptimizationResults(
        mthd, individual, value(st),
        opts.iterations, opts.show_trace, opts.store_trace,
        Evolutionary.metrics(mthd), tr, f_calls(objfun),
        1.0, 1.0, opts.show_trace,
    )
    @test summary(res) == summary(mthd)
    @test Evolutionary.minimizer(res) == individual
    @test minimum(res) == value(st)
    @test Evolutionary.iterations(res) == opts.iterations
    @test Evolutionary.iteration_limit_reached(res) == opts.show_trace
    @test Evolutionary.converged(res) == opts.store_trace
    @test f_calls(res) == f_calls(objfun)
    @test Evolutionary.time_run(res) == 1.0
    @test Evolutionary.time_limit(res) == 1.0
    @test Evolutionary.is_moo(res) == opts.show_trace
    @test length(Evolutionary.trace(res)) >= 1
    @test !Evolutionary.converged(res.metrics[1])
    show(IOBuffer(), res)


    ##############
    # POPULATION #
    ##############
    pop = Evolutionary.initial_population(mthd, BitVector(ones(dimension)))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,)

    @test_throws AssertionError Evolutionary.initial_population(mthd, fill(BitVector(ones(dimension)), 4))
    pop = Evolutionary.initial_population(mthd, fill(BitVector(ones(dimension)), 5))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,)

    pop = Evolutionary.initial_population(mthd, BitMatrix(ones(dimension, 6)))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension, 6)

    pop = Evolutionary.initial_population(mthd, (() -> rand(Bool, dimension)))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,)

    lb = [0, 1, 2, -1]
    ub = [0, 3, 2, 1]
    cb = Evolutionary.ConstraintBounds(lb, ub, [], [])
    pop = Evolutionary.initial_population(mthd, cb)
    @test length(pop) == population_size(mthd)
    @test map(i -> i[1], pop) == fill(ub[1], population_size(mthd))
    @test all(map(i -> lb[2] <= i[2] <= ub[2], pop))
    @test map(i -> i[3], pop) == fill(ub[3], population_size(mthd))
    @test all(map(i -> lb[4] <= i[4] <= ub[4], pop))


    ###########
    # OPTIONS #
    ###########
    opts = Evolutionary.Options(; store_trace = true, iterations = 7)
    @test opts.store_trace
    @test opts.iterations == 7


    ###############
    # CONVERGENCE #
    ###############
    cm = Evolutionary.AbsDiff()
    @test !Evolutionary.converged(cm)
    st.fitness = 1.0
    @test !Evolutionary.assess!(cm, st)
    @test !Evolutionary.converged(cm)
    @test Evolutionary.assess!(cm, st)
    @test Evolutionary.converged(cm)


    ###############
    # CONSTRAINTS #
    ###############
    c = Evolutionary.NoConstraints()
    @test isfeasible(c, 1)
    @test isa(c.bounds, Evolutionary.ConstraintBounds)
    @test constraint_values(c, 1) === nothing
    @test apply!(c, 1) == 1
    @test penalty(c, 1) == 0

    c = BoxConstraints(lb, ub)
    @test isfeasible(c, [0, 1, 2, 0])
    @test constraint_values(c, [1, 4, 2, -2]) === nothing
    @test apply!(c, [1, 4, 2, -2]) == [0, 3, 2, -1]
    @test penalty(c, [1, 4, 2, -2]) == 0

    con_c!(x) = [sum(x)]
    cb = Evolutionary.ConstraintBounds(fill(0, 3), fill(1, 3), [1], [1])
    c = PenaltyConstraints(1, cb, con_c!)
    objfun = EvolutionaryObjective(sum, zeros(3))
    x, y = [0, 1, 0], [0, -1, 3]
    @test isfeasible(c, x)    # feasible
    @test !isfeasible(c, y) # not feasible
    @test apply!(c, x) == x
    @test apply!(c, y) == y
    @test constraint_values(c, x) == [sum(x)]
    @test constraint_values(c, y) == [sum(y)]
    @test penalty(c, x) == 0.0 # c penalty
    @test penalty(c, y) == 1 + 2^2 + 1.0 # c penalty
    @test penalty!([1, 2], c, [x, y]) == [1, 2 + penalty(c, y)]

    struct GenericConstraint <: Evolutionary.AbstractConstraints
        constraint_bounds::Evolutionary.ConstraintBounds{Float64}
    end
    Evolutionary.bounds(c::GenericConstraint) = c.constraint_bounds
    Evolutionary.constraint_values(::GenericConstraint, x) = [sum(x)]

    generic_constraint = GenericConstraint(
        Evolutionary.ConstraintBounds(Float64[], Float64[], [1.0], [1.0])
    )
    @test Evolutionary.constraint_values(generic_constraint, [1.0, 0.0]) == [1.0]
    @test Evolutionary.isfeasible(generic_constraint, [1.0, 0.0])
    @test !Evolutionary.isfeasible(generic_constraint, [0.0, 0.0])


    ############
    # OPTIMIZE #
    ############
    res = Evolutionary.optimize(sum, (() -> BitVector(rand(Bool, dimension))), mthd, opts)
    @test Evolutionary.minimum(res) <= dimension
    @test length(Evolutionary.minimizer(res)) == dimension
    @test Evolutionary.iterations(res) == 7
    @test !Evolutionary.converged(res)
    @test length(Evolutionary.trace(res)) > 1
    @test length(res.metrics) == 1
    @test res.metrics[1].tol == 1.0e-10

    o = Evolutionary.Options(abstol = 1.0e-3)
    res = Evolutionary.optimize(sum, (() -> BitVector(rand(Bool, dimension))), mthd, o)
    @test length(res.metrics) == 1
    @test res.metrics[1].tol == 1.0e-3

    o = Evolutionary.Options(abstol = 1.0e-3, reltol = 1.0e-4)
    res = Evolutionary.optimize(sum, (() -> BitVector(rand(Bool, dimension))), mthd, o)
    @test length(res.metrics) == 2
    @test res.metrics[1].tol == 1.0e-3
    @test res.metrics[2].tol == 1.0e-4

end
