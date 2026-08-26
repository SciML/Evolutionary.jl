using Evolutionary
using NLSolversBase: f_calls, value, value!, value!!
using Test

@testset "Evolutionary Objective" begin

    v = Evolutionary.default_values(ones(3))
    @test eltype(v) == Float64
    @test all(isnan.(v))

    v = Evolutionary.default_values(fill(true, 3))
    @test eltype(v) == Bool
    @test iszero(v)

    v = Evolutionary.default_values(BitVector(ones(Bool, 3)))
    @test eltype(v) == Bool
    @test iszero(v)

    v = Evolutionary.default_values(ones(Int, 2, 3))
    @test eltype(v) == Int
    @test iszero(v)

    f1(x) = sum(x)
    objfun = EvolutionaryObjective(f1, [1; 2; 3; 4])
    @test typeof(objfun.F) == Int
    @test eltype(objfun.x_f) == Int
    f2(x) = sum(x) / length(x)
    objfun = EvolutionaryObjective(f2, [1; 2; 3; 4])
    @test typeof(objfun.F) == Float64
    @test eltype(objfun.x_f) == Int
    objfun = EvolutionaryObjective(f1, BitVector(ones(10)))
    @test typeof(objfun.F) == Int
    @test eltype(objfun.x_f) == Bool
    objfun = EvolutionaryObjective(f2, BitVector(ones(10)))
    @test typeof(objfun.F) == Float64
    @test eltype(objfun.x_f) == Bool
    f3(expr::Expr) = Evolutionary.Expression(expr).([1; 2; 3; 4]) |> sum
    objfun = EvolutionaryObjective(f3, Expr(:call, *, :x, :x))
    @test typeof(objfun.F) == Int
    @test typeof(objfun.x_f) == Expr

    f4(x) = x[1] + x[2]
    x = zeros(2)
    objfun = EvolutionaryObjective(f4, x)
    @test objfun.F isa Float64
    @test objfun.x_f isa Vector{Float64}
    @test !ismultiobjective(objfun)
    @test objfun.F === 0.0
    @test all(isnan.(objfun.x_f))
    v = value(objfun, ones(2))
    @test v == 2.0
    @test v isa Float64
    @test f_calls(objfun) == 1
    @test objfun.F === 0.0
    @test all(isnan.(objfun.x_f))
    @test_throws MethodError value(objfun, ones(Int, 2))
    v = value!(objfun, ones(2))
    @test v == 2.0
    @test v isa Float64
    @test f_calls(objfun) == 2
    @test objfun.F === 2.0
    @test all(isnan.(objfun.x_f))
    v = value!!(objfun, zeros(2))
    @test v === 0.0
    @test v isa Float64
    @test f_calls(objfun) == 3
    @test objfun.F === 0.0
    @test all(iszero.(objfun.x_f))

    function f5(F, x)
        F[1] = x[1] + 1
        F[2] = x[2] + 2
        F
    end
    objfun = EvolutionaryObjective(f5, x, copy(x))
    @test objfun.F isa Vector{Float64}
    @test objfun.x_f isa Vector{Float64}
    @test ismultiobjective(objfun)
    @test_throws MethodError value(objfun, ones(Int, 2))
    @test iszero(objfun.F)
    @test all(isnan.(objfun.x_f))
    v = value(objfun, ones(2))
    @test eltype(v) == Float64
    @test v == [2.0, 3.0]
    @test f_calls(objfun) == 1
    @test iszero(objfun.F)
    @test all(isnan.(objfun.x_f))
    y = zeros(2)
    v = value(objfun, y, ones(2))
    @test v == [2.0, 3.0]
    @test v == y
    @test f_calls(objfun) == 2
    @test iszero(objfun.F)
    @test all(isnan.(objfun.x_f))
    y = zeros(2)
    v = value!(objfun, y, ones(2))
    @test v == [2.0, 3.0]
    @test v == y
    @test f_calls(objfun) == 3
    @test objfun.F == v
    @test all(isnan.(objfun.x_f))
    y = zeros(2)
    v = value!!(objfun, y, zeros(2))
    @test v == [1.0, 2.0]
    @test v == y
    @test f_calls(objfun) == 4
    @test objfun.F == v
    @test iszero(objfun.x_f)

    f6(x) = [x[1], x[2] + 1]
    x = zeros(2)
    objfun = EvolutionaryObjective(f6, x)
    v = value(objfun, ones(2))
    @test v == [1.0, 2.0]
    @test iszero(objfun.F)
    @test all(isnan.(objfun.x_f))
    v = value!(objfun, ones(2))
    @test v == [1.0, 2.0]
    @test objfun.F == v
    @test all(isnan.(objfun.x_f))
    v = value!!(objfun, zeros(2))
    @test objfun.F == v
    @test iszero(objfun.x_f)

    X = [ones(2) for i in 1:10]
    F = zeros(10)
    objfun = EvolutionaryObjective(f4, x)
    V = value!(objfun, F, X)
    @test F == 2ones(10)

    objfun = EvolutionaryObjective(f4, x; eval = :thread)
    V = value!(objfun, F, X)
    @test F == 2ones(10)

    objfun = EvolutionaryObjective(f5, x, zeros(2))
    F = zeros(2, 10)
    V = value!(objfun, F, X)
    @test F == [2ones(10) 3ones(10)]'

    objfun = EvolutionaryObjective(f5, x, zeros(2); eval = :thread)
    F = zeros(2, 10)
    V = value!(objfun, F, X)
    @test F == [2ones(10) 3ones(10)]'

end
