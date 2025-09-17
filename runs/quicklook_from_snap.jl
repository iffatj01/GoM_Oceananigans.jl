# runs/quicklook_from_snap.jl

using JLD2
using CairoMakie            # if missing: import Pkg; Pkg.add("CairoMakie")
using Printf

# ----- paths -----
runs_dir = @__DIR__
root_dir = normpath(joinpath(runs_dir, ".."))
outdir   = joinpath(root_dir, "outputs")
gridfile = joinpath(root_dir, "data", "processed", "ibgrid_gom.jld2")

isdir(outdir) || error("No output folder found at $outdir")

# ----- newest snapshot -----
snaps = sort(filter(f -> occursin(r"^snap_\d+\.jld2$", f), readdir(outdir)))
isempty(snaps) && error("No snap_*.jld2 files found in $outdir")
snapf = joinpath(outdir, last(snaps))
@info "Reading newest snapshot" file=snapf

# ----- read fields -----
lon = lat = zc = nothing
T = η = u = v = nothing
time = nothing

jldopen(snapf, "r") do f
    haskey(f, "lon") && (lon = read(f, "lon"))
    haskey(f, "lat") && (lat = read(f, "lat"))
    haskey(f, "zc")  && (zc  = read(f, "zc"))

    for nm in ("T_cpu", "T");                 haskey(f, nm) && (T = read(f, nm); break); end
    for nm in ("η_cpu","eta_cpu","η","eta");  haskey(f, nm) && (η = read(f, nm); break); end
    for nm in ("u_cpu", "u");                 haskey(f, nm) && (u = read(f, nm); break); end
    for nm in ("v_cpu", "v");                 haskey(f, nm) && (v = read(f, nm); break); end
    for nm in ("time","t");                   haskey(f, nm) && (time = read(f, nm); break); end
end

# ----- lon/lat fallback built from metadata (avoid loading heavy grid types) -----
if lon === nothing || lat === nothing
    @info "lon/lat not stored in snapshot; building from ibgrid_gom.jld2 metadata."
    isfile(gridfile) || error("Coordinates missing in snapshot and $gridfile not found.")
    # Only load simple scalars we saved during preprocess:
    lonW = lonE = latS = latN = NaN
    Nx = Ny = Nz = 0
    Hmax = NaN
    @load gridfile lonW lonE latS latN Nx Ny Nz Hmax
    (isfinite(lonW) && isfinite(lonE) && Nx > 0 && Ny > 0) ||
        error("Missing lon/lat bounds or sizes in $gridfile")
    lon = collect(range(lonW, lonE; length=Int(Nx)))
    lat = collect(range(latS, latN; length=Int(Ny)))
end

@assert T !== nothing "Temperature field not found in $snapf"
@assert η !== nothing "Free-surface η not found in $snapf"

# ----- quicklooks -----
Tsurf = @views T[:, :, end]
speed = (u !== nothing && v !== nothing) ? @views sqrt.(u[:, :, end].^2 .+ v[:, :, end].^2) : nothing

# Surface temperature
fig1 = Figure(resolution = (1000, 650))
ax1  = Axis(fig1[1,1], xlabel="Longitude (°E)", ylabel="Latitude (°N)",
            title = @sprintf("Surface T  (t = %.2f days)", time === nothing ? NaN : time/86400))
hm1  = heatmap!(ax1, lon, lat, Tsurf'; interpolate=false)
Colorbar(fig1[1,2], hm1, label="°C")
save(joinpath(outdir, "quicklook_Tsurf.png"), fig1)

# Sea-surface height
fig2 = Figure(resolution = (1000, 650))
ax2  = Axis(fig2[1,1], xlabel="Longitude (°E)", ylabel="Latitude (°N)",
            title = @sprintf("Free surface η  (t = %.2f days)", time === nothing ? NaN : time/86400))
hm2  = heatmap!(ax2, lon, lat, η'; interpolate=false)
Colorbar(fig2[1,2], hm2, label="m")
save(joinpath(outdir, "quicklook_eta.png"), fig2)

# Optional: surface current speed
if speed !== nothing
    fig3 = Figure(resolution = (1000, 650))
    ax3  = Axis(fig3[1,1], xlabel="Longitude (°E)", ylabel="Latitude (°N)",
                title = @sprintf("Surface speed |u|  (t = %.2f days)", time === nothing ? NaN : time/86400))
    hm3  = heatmap!(ax3, lon, lat, speed'; interpolate=false)
    Colorbar(fig3[1,2], hm3, label="m s⁻¹")
    save(joinpath(outdir, "quicklook_speed.png"), fig3)
end

println("Saved to $outdir:")
println("  quicklook_Tsurf.png")
println("  quicklook_eta.png")
if speed !== nothing
    println("  quicklook_speed.png")
end
