"""
Abstract evolutionary optimization result type
"""
abstract type OptimizationResults end

"""
    summary(result)

Shows the optimization algorithm that produced this `result`.
"""
summary(or::OptimizationResults) = summary(or.method)

"""
    minimizer(result)

A minimizer object from the optimization `result`.
"""
minimizer(r::OptimizationResults) = r.minimizer

"""
    minimum(result)

A minimum value from the optimization `result`.
"""
minimum(r::OptimizationResults) = r.minimum

"""
    iterations(result)

A number of iterations to reach the minimum.
"""
iterations(r::OptimizationResults) = r.iterations

"""
    iteration_limit_reached(result)

Returns `true` if the iteration limit was reached.
"""
iteration_limit_reached(r::OptimizationResults) = r.iteration_converged

"""
    trace(result)

Returns a trace of optimization states from the optimization `result`.
"""
trace(r::OptimizationResults) = length(r.trace) > 0 ? r.trace : error("No trace in optimization results. To get a trace, run optimize() with store_trace = true.")

"""
    f_calls(result)

Returns a number of an objective function calls.
"""
f_calls(r::OptimizationResults) = r.f_calls

"""
    abstol(result)

Returns an absolute tolerance value of the optimization `result`.
"""
abstol(r::OptimizationResults) = error("`abstol` is not implemented for $(summary(r)).")

"""
    reltol(result)

Returns a relative tolerance value of the optimization `result`.
"""
reltol(r::OptimizationResults) = error("`reltol` is not implemented for $(summary(r)).")

abschange(r::OptimizationResults) = error("`abschange` is not implemented for $(summary(r)).")
relchange(r::OptimizationResults) = error("`relchange` is not implemented for $(summary(r)).")


"""
Evolutionary optimization result type
"""
mutable struct EvolutionaryOptimizationResults{O<:AbstractOptimizer, Tx, Tf} <: OptimizationResults
    method::O
    minimizer::Tx
    minimum::Tf
    iterations::Int
    iteration_converged::Bool
    converged::Bool
    metrics::ConvergenceMetrics
    trace::OptimizationTrace
    f_calls::Int
    time_limit::Float64
    time_run::Float64
    is_moo::Bool
end

"""
    converged(result)

Returns `true` if the optimization successfully converged to a minimum value.
"""
converged(r::EvolutionaryOptimizationResults) = r.converged
time_limit(r::EvolutionaryOptimizationResults) = r.time_limit
time_run(r::EvolutionaryOptimizationResults) = r.time_run
is_moo(r::EvolutionaryOptimizationResults) = r.is_moo

function show(io::IO, r::EvolutionaryOptimizationResults)
    failure_string = "failure"
    if iteration_limit_reached(r)
        failure_string *= " (reached maximum number of iterations)"
    end
    if time_run(r) > time_limit(r)
        failure_string *= " (exceeded time limit of $(time_limit(r)))"
    end
    print(io, "\n")
    print(io, " * Status: ", converged(r) ? "success" : failure_string, "\n\n")
    print(io, " * Candidate solution\n")
    mzr = minimizer(r)
    if is_moo(r)
        pfsize = length(mzr)
        pl = pfsize > 1 ? "s" : ""
        print(io, "    Pareto front: $(pfsize) element$pl\n")
    elseif mzr isa AbstractVector
        nx = length(mzr)
        str_x_elements = ["$_x" for _x in Iterators.take(mzr, min(nx, 3))]
        if nx >= 4
            push!(str_x_elements, " ...")
        end
        print(io, "    Minimizer:  [", join(str_x_elements, ", "),  "]\n")
    else
        print(io, "    Minimizer:  $mzr\n")
    end
    !is_moo(r) && print(io, "    Minimum:    $(minimum(r))\n")
    print(io, "    Iterations:", rpad(" ",is_moo(r) ? 3 : 0), "$(iterations(r))\n")
    print(io, "\n")
    print(io, " * Found with\n")
    print(io, "    Algorithm: $(summary(r))\n")
    print(io, "\n")
    if length(r.metrics) > 0
        print(io, " * Convergence measures\n")
        maxdsclen = maximum(length(description(cm)) for cm in r.metrics)
        rpd = " "^4
        for cm in r.metrics
            sgn = converged(cm) ? "≤" : "≰"
            dsc = description(cm)
            lpd = " "^(maxdsclen + 1 - length(dsc))
            !isa(cm, Never) && print(io, "$rpd$dsc$lpd= $(diff(cm)) $sgn $(tolerance(cm))\n" )
        end
        print(io, "\n")
    end
    print(io, " * Work counters\n")
    tr = round(time_run(r); digits=4)
    tl = isnan(time_limit(r)) ? Inf : round(time_limit(r); digits=4)
    print(io, "    Seconds run:   $tr (vs limit $tl)\n")
    print(io, "    Iterations:    $(iterations(r))\n")
    print(io, "    f(x) calls:    $(f_calls(r))\n")
    return
end

