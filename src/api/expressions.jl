"""
Julia expression wrapper
"""
struct Expression
    expr::Expr
    syms::Dict{Symbol, Int}
end
function Expression(ex::Expr)
    syms = Dict(s => i for (i, s) in pairs(sort!(symbols(ex))))
    return Expression(ex, syms)
end
show(io::IO, e::Expression) = infix(io, e.expr)
(e::Expression)(vals...) = evaluate(e.expr, e.syms, vals...)

function symbols(ex::Expr)
    syms = Symbol[]
    for e in ex.args
        isa(e, Symbol) && push!(syms, e)
        isa(e, Expr) && append!(syms, symbols(e))
    end
    return unique!(syms)
end

function compile(ex::Expr, params::Vector{Symbol})
    tprm = Expr(:tuple, params...)
    return Expr(:->, tprm, ex)
end

height(ex) = isa(ex, Expr) ? maximum(height(e) for e in ex.args) + 1 : 0
nodes(ex) = !isa(ex, Expr) ? 1 : length(ex.args) > 0 ? sum(nodes(e) for e in ex.args) : 0
subexpression_count(ex) =
    !isa(ex, Expr) ? 1 : 1 + sum(subexpression_count(e) for e in ex.args[2:end])

function depth(root, ex; d = 0)
    return if root == ex
        d
    elseif isa(root, Expr)
        maximum(depth(e, ex, d = d + 1) for e in root.args)
    else
        -1
    end
end

function copy_expression!(dst::Expr, src::Expr)
    dst.head = src.head
    dst.args = deepcopy(src.args)
    return dst
end

function randsubexpr(ex)
    !isa(ex, Expr) && return ex
    csize = map(subexpression_count, ex.args[2:end])
    cidx = rand(1:(sum(csize) + 1)) - 1
    return if cidx == 0
        ex
    else
        nidx = something(
            findfirst(i -> cidx <= i, cumsum(csize)),
            throw(ArgumentError("expression subtree index is invalid"))
        )
        randsubexpr(ex.args[nidx + 1])
    end
end

function subexpression(ex::Expr, idx::Int)
    idx < 0 && throw(BoundsError(ex, idx))
    counter = Ref(0)

    function find_subexpression(node)
        counter[] == idx && return true, node
        counter[] += 1
        node isa Expr || return false, nothing
        for child in node.args[2:end]
            found, subex = find_subexpression(child)
            found && return true, subex
        end
        return false, nothing
    end

    found, subex = find_subexpression(ex)
    found || throw(BoundsError(ex, idx))
    return subex
end

function replace_subexpression!(ex::Expr, subex, idx::Int)
    idx <= 0 && throw(BoundsError(ex, idx))
    counter = Ref(0)

    function replace_in_children!(node::Expr)
        for i in 2:length(node.args)
            counter[] += 1
            if counter[] == idx
                node.args[i] = subex
                return true
            elseif node.args[i] isa Expr && replace_in_children!(node.args[i])
                return true
            end
        end
        return false
    end

    replace_in_children!(ex) || throw(BoundsError(ex, idx))
    return ex
end

isnum(ex) = isa(ex, Number)
iszeronum(root) = isnum(root) && iszero(root)
isonenum(root) = isnum(root) && isone(root)
isdiv(ex) = ex in [/, div, pdiv, aq]
isexpr(ex) = isa(ex, Expr)
issym(ex) = isa(ex, Symbol)
isexprsym(ex) = isexpr(ex) || issym(ex)
isbinexpr(ex) = isexpr(ex) && length(ex.args) == 3

function evaluate(ex::Expr, psyms::Dict{Symbol, Int}, vals...)
    exprm = ex.args
    exvals = (isexpr(nex) || issym(nex) ? evaluate(nex, psyms, vals...) : nex for nex in exprm[2:end])
    return try
        exprm[1](exvals...)
    catch err
        @error "Incorrect expression" ex psyms vals
        rethrow(err)
    end
end

function evaluate(ex::Symbol, psyms::Dict{Symbol, Int}, vals...)
    pidx = get(psyms, ex, 0)
    if pidx == 0
        @error "Undefined symbol: $ex"
    end
    return vals[pidx]
end

function simplifyunary!(root)
    fn, op = root.args
    if isexpr(op)
        fn2, op2 = op.args
        if fn == log && fn2 == exp
            return op2
        elseif fn == exp && fn2 == log
            return op2
        end
    end
    return root
end

function simplifybinary!(root)
    fn, op1, op2 = root.args

    # some elementary simplification rules
    if fn == (-) && op1 == op2 # additive inverse: x-x = 0
        root = 0
    elseif isdiv(fn) && (op1 == op2 || iszeronum(op2)) # look for multiplicative inverse x/x = 1
        root = 1
    elseif (fn == (*) || isdiv(fn)) &&  (iszeronum(op1) || iszeronum(op2)) # look for 0: x*0 = 0*x = 0
        root = 0
    elseif (fn == (+) || fn == (-)) && iszeronum(op2) # look for 0: x ± 0 = x
        root = op1
    elseif fn == (+) && iszeronum(op1) # look for 0: 0 + x = x
        root = op2
    elseif (fn == (*) || isdiv(fn)) && isonenum(op2) # x*1 = x || x/1 = x
        root = op1
    elseif fn == (*) && isonenum(op1) # 1*x = x
        root = op2
    elseif (fn == (+)) && (op1 == op2) # x+x = 2x
        root.args[1] = (*)
        root.args[2] = 2
    elseif (fn == (+) || fn == (-)) && (isbinexpr(op1) && isnum(op2))
        fn2, op11, op12 = op1.args
        if fn == (+) && fn2 == (-) && isnum(op12)
            # (x-m)+n = x+(n-m)
            root.args[2] = op11
            root.args[3] = op2 - op12
        elseif fn == (-) && fn2 == (-) && isnum(op12)
            # (x-m)-n = x-(n+m)
            root.args[2] = op11
            root.args[3] = op12 + op2
        elseif fn2 == (+) && (isnum(op11) || isnum(op12))
            # (m+x)±n = (x+m)±n = x+(n±m)
            var, n2 = isnum(op11) ? (op12, op11) : (op11, op12)
            n3 = fn(n2, op2)
            root.args[1] = fn2
            root.args[2] = var
            root.args[3] = n3
        elseif fn2 == (-) && isnum(op11)
            # (m-x)±n = (n±m)-x
            n3 = fn(op11, op2)
            root.args[1] = fn2
            root.args[2] = n3
            root.args[3] = op12
        end
    elseif fn == (+) && isnum(op1) && isbinexpr(op2)
        fn2, op21, op22 = op2.args
        if fn2 == (+) || fn2 == (-)
            if isnum(op21)
                # n+(m±x) = (n+m)±x
                root.args[1] = fn2
                root.args[2] = op1 + op21
                root.args[3] = op22
            elseif isnum(op22)
                # n+(x±m) = (n±m)+x
                root.args[1] = fn
                root.args[2] = fn2(op1, op22)
                root.args[3] = op21
            end
        end
    elseif fn == (-) && isnum(op1) && isbinexpr(op2)
        fn2, op21, op22 = op2.args
        if fn2 == (+) && (isnum(op21) || isnum(op22))
            # n-(x+m) = n-(m+x) = (n-m)-x
            var, n2 = isnum(op21) ? (op22, op21) : (op21, op22)
            root.args[2] = op1 - n2
            root.args[3] = var
        elseif fn2 == (-) && (isnum(op21) || isnum(op22))
            # n-(m-x) = (n-m)+x
            # n-(x-m) = (n+m)-x
            var, n2, f1p = isnum(op21) ? (op22, op21, true) : (op21, op22, false)
            root.args[1] = f1p ? (+) : (-)
            root.args[2] = f1p ? op1 - n2 : op1 + n2
            root.args[3] = var
        end
    elseif fn == (+) && (isexpr(op1) || isexpr(op2))
        if isbinexpr(op2)
            # x + (n - x) = n
            fn2, op21, op22 = op2.args
            if fn2 == (-) && op1 == op22
                root = op21
            end
        elseif isbinexpr(op1)
            # (n - x) + x = n
            fn2, op11, op12 = op1.args
            if fn2 == (-) && op12 == op2
                root = op11
            end
        end
    elseif fn == (-) && (isexpr(op1) || isexpr(op2))
        if isbinexpr(op2)
            fn2, op21, op22 = op2.args
            if op1 == op21
                # x - (x ± n) = -±n
                if fn2 == (-)
                    root = op22
                elseif fn2 == (+)
                    pop!(root.args)
                    root.args[end] = op22
                end
            elseif op1 == op22 && fn2 == (+)
                # x - (n + x) = -n
                pop!(root.args)
                root.args[end] = op21
            end
        elseif isbinexpr(op1)
            # (x ± n) - x = ±n
            fn2, op11, op12 = op1.args
            if op11 == op2
                if fn2 == (+)
                    root = op12
                elseif fn2 == (-)
                    root.args[1] = fn2
                    root.args[2] = op12
                    pop!(root.args)
                end
            end
        end
    end

    # evaluate numerical arithmetic
    isnum(op1) && isnum(op2) && return fn(op1, op2)

    return root
end

"""
    simplify!(expr)

Simplify an algebraic expression
"""
function simplify!(root)
    # return if not an expression
    !isa(root, Expr) && return root
    # return if not a function
    root.head != :call && return root

    n = length(root.args)
    # recursively simplify arguments
    for i in 2:n
        !isexpr(root.args[i]) && continue
        root.args[i] = simplify!(root.args[i])
    end

    n == 2 && return simplifyunary!(root)
    n == 3 && return simplifybinary!(root)
    return root
end

# This doesn't extend `Base.contains` because that would be piracy
function contains(ex::Expr, sym::Symbol)
    for arg in ex.args
        issym(arg) && arg == sym && return true
        isa(arg, QuoteNode) && arg.value == sym && return true
        isa(arg, Function) && Symbol(arg) == sym && return true
        isexpr(arg) && return contains(arg, sym)
    end
    return false
end

# Construct infix string from an expression
function infix(io::IO, root; digits = 3)
    return if isa(root, Number)
        print(io, round(root, digits = digits))
    elseif isa(root, Expr) && root.head == :call
        if root.args[1] ∈ [+, -, *, /, ^]
            print(io, "(")
            infix(io, root.args[2])
            infix(io, root.args[1])
            length(root.args) > 2 && infix(io, root.args[3])
            print(io, ")")
        else
            infix(io, root.args[1])
            print(io, "(")
            for (i, ex) in enumerate(root.args[2:end])
                i > 1 && print(io, ", ")
                infix(io, ex)
            end
            print(io, ")")
        end
    else
        print(io, root)
    end
end
show(io::IO, ::MIME"text/html", e::Expression) = infix(io, e.expr)

latex(io::IO, root::Number; digits = 3) = print(io, round(root, digits = digits))
latex(io::IO, root::Symbol; kwargs...) = print(io, root)
function latex(io::IO, root::Expr; digits = 3)
    root.head != :call && return print(io, root)
    return if root.args[1] == (/)
        print(io, "\\frac{")
        latex(io, root.args[2], digits = digits)
        print(io, "}{")
        latex(io, root.args[3], digits = digits)
        print(io, "}")
    elseif root.args[1] ∈ [+, -, *]
        print(io, "\\left(")
        latex(io, root.args[2], digits = digits)
        print(io, root.args[1])
        latex(io, root.args[3], digits = digits)
        print(io, "\\right)")
    elseif root.args[1] == (^)
        print(io, "{")
        latex(io, root.args[2], digits = digits)
        print(io, "}^{")
        latex(io, root.args[3], digits = digits)
        print(io, "}")
    else
        print(io, "\\")
        print(io, root.args[1])
        print(io, "\\left(")
        for ex in root.args[2:end]
            latex(io, ex, digits = digits)
        end
        print(io, "\\right)")
    end
end
show(io::IO, ::MIME"text/latex", e::Expression) = latex(io, e.expr)
