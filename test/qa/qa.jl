using SciMLTesting, Evolutionary, Test

run_qa(
    Evolutionary;
    explicit_imports = true,
    # ambiguities, unbound_args and piracies still fail; keep them disabled and
    # tracked. https://github.com/SciML/Evolutionary.jl/issues/145
    aqua_broken = (
        :ambiguities,    # 18 ambiguities in optimize overloads (src/api/optimize.jl)
        :unbound_args,   # 3 methods (src/api/expressions.jl)
        :piracies,       # 7 methods (Expr ops, replace, value)
    ),
    ei_kwargs = (;
        # other packages' non-public names: ignore until they go public upstream.
        all_explicit_imports_are_public = (;
            ignore = (
                :default_rng,     # Random (stdlib non-public)
                :nconstraints,    # NLSolversBase non-public
                :nconstraints_x,  # NLSolversBase non-public
            ),
        ),
    ),
)

# JET surfaces method/typo errors only on Julia >= 1.12 (the QA `1` lane), while the
# LTS lane is clean, so neither a hard JET check (fails 1.12) nor `jet_broken = true`
# (auto-flags an Unexpected Pass on the clean LTS lane) is correct across the matrix.
# Keep the finding tracked as a static broken placeholder.
# https://github.com/SciML/Evolutionary.jl/issues/145
@testset "JET" begin
    @test_broken false  # report_package: +(::Nothing,::Int), kwcall on SPX, Evolutionary.expr (Julia >= 1.12)
end
