using JLD2

outdir = normpath(joinpath(@__DIR__, "..", "outputs"))
snaps = sort(filter(f -> occursin(r"^snap_\d+\.jld2$", f), readdir(outdir)))
isempty(snaps) && error("No snap_*.jld2 files in $outdir")
snap = joinpath(outdir, last(snaps))
println("Inspecting: ", snap)

# Try the high-level loader (variables saved with @save show up here)
vars = JLD2.load(snap)
println("Top-level variable names: ", join(string.(keys(vars)), ", "))

# Also walk the JLD2 groups to see everything stored
jldopen(snap, "r") do f
    println("\n=== JLD2 tree ===")
    function walk(g, prefix="")
        for k in keys(g)
            path = isempty(prefix) ? k : string(prefix, "/", k)
            x = g[k]
            println(path, " :: ", typeof(x))
            if x isa JLD2.Group
                walk(x, path)
            end
        end
    end
    walk(f)
end
