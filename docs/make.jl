using ModalDecisionLists
using Documenter

DocMeta.setdocmeta!(ModalDecisionLists, :DocTestSetup, :(using ModalDecisionLists); recursive=true)

makedocs(;
    modules=[ModalDecisionLists, ModalDecisionLists.MLJInterface, ModalDecisionLists.LossFunctions],
    authors="Giovanni Pagliarini, Edoardo Ponsanesi",
    repo=Documenter.Remotes.GitHub("aclai-lab", "ModalDecisionLists.jl"),
    sitename="ModalDecisionLists.jl",
    format=Documenter.HTML(;
        size_threshold = 4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aclai-lab.github.io/ModalDecisionLists.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly = true, # TODO remove?
)

deploydocs(;
    repo = "github.com/aclai-lab/ModalDecisionLists.jl",
    devbranch = "main",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
