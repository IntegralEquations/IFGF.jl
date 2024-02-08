using Documenter
using IFGF

draft = true

const ON_CI = get(ENV, "CI", "false") == "true"

ON_CI && (draft = false) # always full build on CI

println("\n*** Generating documentation")

DocMeta.setdocmeta!(IFGF, :DocTestSetup, :(using IFGF); recursive = true)

makedocs(;
    modules = modules,
    repo = "",
    sitename = "IFGF.jl",
    format = Documenter.HTML(;
        prettyurls = ON_CI,
        canonical = "https://github.com/IntegralEquations/IFGF.jl",
        assets = String[],
    ),
    pages = [
        # "Home" => "index.md",
        "References" => "references.md",
    ],
    warnonly = ON_CI ? false : Documenter.except(:linkcheck_remotes),
    pagesonly = true,
    draft,
)

deploydocs(;
    repo = "https://github.com/IntegralEquations/IFGF.jl",
    devbranch = "main",
    push_preview = true,
)
