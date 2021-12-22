using IFGF
using Documenter

DocMeta.setdocmeta!(IFGF, :DocTestSetup, :(using IFGF); recursive=true)

makedocs(;
    modules=[IFGF],
    authors="Luiz M. Faria <maltezfaria@gmail.com> and contributors",
    repo="https://github.com/WaveProp/IFGF.jl/blob/{commit}{path}#{line}",
    sitename="IFGF.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://WaveProp.github.io/IFGF.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/WaveProp/IFGF.jl",
    devbranch="main"
)
