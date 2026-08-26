```@setup cntr
using Evolutionary
```

The nonlinear constrained optimization interface in `Evolutionary` assumes that the user can write the optimization problem as follows:

```math
\min_{x\in\mathbb{R}^n} f(x) \quad \text{such that}\\
l_x \leq \phantom{c(}x\phantom{)} \leq u_x \\
l_c \leq c(x) \leq u_c.
```

For equality constraints on ``x_j`` or ``c(x)_j`` you set those particular entries of bounds to be equal, ``l_j=u_j``.
Likewise, setting ``l_j=-\infty`` or ``u_j=\infty`` means that the  constraint is unbounded from below or above respectively.

Following examples show how to use constraints to optimize the [Booth function](https://www.sfu.ca/~ssurjano/booth.html).
The function is defined as follows:

```@repl cntr
f(x)=(x[1]+2x[2]-7)^2+(2x[1]+x[2]-5)^2 # Booth
```

The function is usually evaluated on the square ``x_i ∈ [-10, 10]``, for all ``i = 1, 2``.
The global minimum on this function is located at ``(1,3)``.

```@example cntr
ga = GA(populationSize=100,selection=uniformranking(3),
        mutation=gaussian(),crossover=uniformbin())
x0 = [0., 0.]
results = Evolutionary.optimize(f, x0, ga)
```

## Box Constrained Optimization

We want to optimize the [Booth function](https://www.sfu.ca/~ssurjano/booth.html) in
the box ``0.5 \leq x_i \leq 2.0``, starting from the point ``x_0=(1,1)``.

Reusing our Booth example from above, boxed minimization is performed by providing
vectors of lower and upper bounds as follows:

```@example cntr
lower = [0.5, 0.5]
upper = [2.0, 2.0]
x0 = [1., 1.]
results = Evolutionary.optimize(f, BoxConstraints(lower, upper), x0, ga)
```

The box constraints can be also defined using [`BoxConstraints`](@ref) object.

```@docs
BoxConstraints
```

```@example cntr
cnst = BoxConstraints([0.5, 0.5], [2.0, 2.0])
```

This object can be passed as a parameter to the optimization call, [`Evolutionary.optimize`](@ref):

```@example cntr
results = Evolutionary.optimize(f, cnst, x0, ga) # or Evolutionary.optimize(f, cnst, ga)
```

## Penalty Constrained Optimization

For the penalty constrained optimization, any value and linear/nonlinear constraints
are transformed into the penalty to the minimized fitness function.
In order to provide linear/nonlinear constraints to an optimization problem,
you can use the following penalty constraint algorithm:

```@docs
PenaltyConstraints
```

We want to minimize the following function ``f(x,y) = 3x+9y`` that is subject to constraints
``\sqrt(xy) \geq 100`` and ``x,y \geq 0``. The minimum of this function is near ``(173.41, 57.8)``.
We begin  by defining constraints as follows:

```@example cntr
# x, y ≥ 0
lx = [0.0, 0.0] # lower bound for values
ux = [Inf, Inf] # upper bound for values
# √xy ≥ 100
c(x) = [ prod(map(e->sqrt(e>0 ? e : 0.0), x)) ] # negative values are zeroed
lc   = [100.0] # lower bound for constraint function
uc   = [ Inf ]   # upper bound for constraint function
con = PenaltyConstraints(100.0, lx, ux, lc, uc, c) # first parameter is a penalty value
```

Now, we define the fitness function, an initial individual structure, and algorithm parameters;
then we perform minimization as follows:

```@example cntr
f(x) = 3x[1]+9x[2] # fitness function
x0 = [1., 1.] # individual
ga = GA(populationSize=100,selection=tournament(7),
        mutation=gaussian(),crossover=intermediate(2))
results = Evolutionary.optimize(f, con, x0, ga)
```

We can use worst fitness constraint algorithm which doesn't require to specify the constraint penalty value

```@docs
WorstFitnessConstraints
```
```@example cntr
con = WorstFitnessConstraints(lx, ux, lc, uc, c)
```
```@example cntr
results = Evolutionary.optimize(f, con, x0, ga)
```

## [Auxiliary Functions](@id cons_aux_funcs)

```@docs
constraint_values(c::Evolutionary.AbstractConstraints, x)
isfeasible
penalty
penalty!
apply!
bounds
```
