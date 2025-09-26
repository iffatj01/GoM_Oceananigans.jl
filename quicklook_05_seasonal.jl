# runs/quicklook_from_snap.jl
#
# Quicklook plots from the latest snapshot:
#  - Surface temperature (T)
#  - Free-surface height (η)
#  - Surface speed from (u, v), averaged to centers
#
# Requires: JLD2, Plots

using JLD2
using Printf
using Statistics
using Plots

# -------------------------
# Paths
# -------------------------
const ROOT     = normpath(joinpath(@__DIR__, ".."))
const OUTDIR   = joinpath(ROOT, "outputs")
const PROCMETA = joinpath(ROOT, "data", "processed", "ibgrid_gom.jld2")

mkpath(OUTDIR)

# -------------------------
# Utilities
# -------------------------
function newest_snapshot(dir::String)
    files = readdir(dir; join=true)
    # prefer snap_*.jld2, else any *.jld2
    snaps = filter(f -> occursin(r"snap_.*\.jld2$", f), files)
    if isempty(snaps)
        snaps = filter(f -> endswith(f, ".jld2"), files)
    end
    isempty(snaps) && error("No JLD2 files found in $dir")
    sort(snaps)[end]
end

function top_keys(path::String)
    jldopen(path, "r") do f
        # top-level names are the keys of the file
        collect(keys(f))
    end
end

# Load a variable by trying a list of candidate names
function loadvar(path::String, candidates::Vector{String})
    jldopen(path, "r") do f
        for name in candidates
            haskey(f, name) && return read(f[name])
        end
    end
    return nothing
end

# Build lon/lat centers from ibgrid metadata we saved earlier
function lonlat_from_meta(meta_path::String)
    @assert isfile(meta_path) "Metadata file not found: $meta_path"
    Nx = Ny = lonW = lonE = latS = latN = nothing
    jldopen(meta_path, "r") do f
        Nx   = read(f["Nx"])
        Ny   = read(f["Ny"])
        lonW = read(f["lonW"])
        lonE = read(f["lonE"])
        latS = read(f["latS"])
        latN = read(f["latN"])
    end
    Δlon = (lonE - lonW) / Nx
    Δlat = (latN - latS) / Ny
    lonc = collect(range(lonW + Δlon/2, stop=lonE - Δlon/2, length=Nx))
    latc = collect(range(latS + Δlat/2, stop=latN - Δlat/2, length=Ny))
    return lonc, latc, Nx, Ny
end

# Convert a 3D field to its surface (top) 2D slice.
surface2d(A::AbstractArray) = ndims(A) == 3 ? dropdims(@view A[:, :, end]; dims=3) :
                                               ndims(A) == 2 ? A :
                                               error("Unsupported array dims $(ndims(A))")

# Robust color limits based on finite data percentiles
function robust_clims(A::AbstractArray)
    data = vec(A[isfinite.(A)])
    if isempty(data)
        return (0.0, 1.0)
    end
    sort!(data)
    loidx = max(1, Int(floor(0.02*length(data))))
    hiidx = max(loidx, Int(ceil(0.98*length(data))))
    lo, hi = data[loidx], data[hiidx]
    if hi ≈ lo
        ϵ = max(abs(hi), 1e-12) * 1e-3
        return (lo - ϵ, hi + ϵ)
    end
    return (lo, hi)
end

# Face-to-center averaging for C-grid surface slices:
# u is (Nx+1, Ny), v is (Nx, Ny+1) → centers (Nx, Ny)
function average_to_centers(u2::AbstractMatrix, v2::AbstractMatrix)
    (Nu_x, Nu_y) = size(u2)   # should be (Nx+1, Ny)
    (Nv_x, Nv_y) = size(v2)   # should be (Nx,   Ny+1)
    Nx = Nu_x - 1
    Ny = Nu_y
    @assert Nv_x == Nx && Nv_y == Ny + 1 "Cannot reconcile u size $(size(u2)) and v size $(size(v2))"
    u_c = 0.5 .* (u2[1:end-1, :] .+ u2[2:end, :])     # (Nx, Ny)
    v_c = 0.5 .* (v2[:, 1:end-1] .+ v2[:, 2:end])     # (Nx, Ny)
    return u_c, v_c
end

# Save heatmap with tidy layout
function save_heatmap(data::AbstractMatrix; lon::Vector, lat::Vector, title::String, fname::String, clims_auto::Bool=true)
    # Ensure data matches lon×lat layout
    @assert size(data, 1) == length(lon) "data size $(size(data)) vs lon length $(length(lon))"
    @assert size(data, 2) == length(lat) "data size $(size(data)) vs lat length $(length(lat))"
    Z = permutedims(data, (2,1)) # Plots: heatmap(x, y, Z) expects Z[j,i]
    clim = clims_auto ? robust_clims(Z) : nothing
    plt = heatmap(lon, lat, Z;
                  color=:viridis, clims=clim, aspect_ratio=1,
                  xlabel="Longitude (°E)", ylabel="Latitude (°N)",
                  title=title, right_margin=5Plots.mm, left_margin=5Plots.mm,
                  top_margin=5Plots.mm, bottom_margin=5Plots.mm,
                  framestyle=:box, colorbar=true)
    savefig(plt, fname)
    @info "Saved: $fname"
    nothing
end

# -------------------------
# Main
# -------------------------
SNAP = newest_snapshot(OUTDIR)
@info "Reading snapshot" file=SNAP

# Show keys (helps when something is missing)
keys_in_file = top_keys(SNAP)
@info "Top-level keys in file: $(join(keys_in_file, ", "))"

# Load vars with fallbacks
T3  = loadvar(SNAP, ["T_cpu", "T", "temperature", "Θ", "theta"])
ηX  = loadvar(SNAP, ["η_cpu", "eta_cpu", "η", "eta"])
u3  = loadvar(SNAP, ["u_cpu", "u"])
v3  = loadvar(SNAP, ["v_cpu", "v"])

# Coordinates from metadata
lonc, latc, Nx, Ny = lonlat_from_meta(PROCMETA)

# --------- Temperature ---------
if T3 !== nothing
    T2 = surface2d(T3)
    @info @sprintf("T stats (surface): min=%g max=%g mean=%g", minimum(T2), maximum(T2), mean(T2))
    save_heatmap(T2; lon=lonc, lat=latc, title="Surface temperature", fname=joinpath(OUTDIR, "quicklook_T.png"))
else
    @info "Skipping T plot (temperature not found)."
end

# --------- Free surface η ---------
if ηX !== nothing
    η2 = surface2d(ηX)
    @info @sprintf("η stats (surface): min=%g max=%g mean=%g", minimum(η2), maximum(η2), mean(η2))
    save_heatmap(η2; lon=lonc, lat=latc, title="Free-surface height η", fname=joinpath(OUTDIR, "quicklook_eta.png"))
else
    @info "Skipping η plot (η/eta not found)."
end

# --------- Surface speed from u,v ---------
if (u3 !== nothing) && (v3 !== nothing)
    u2 = surface2d(u3)  # (Nx+1, Ny) expected
    v2 = surface2d(v3)  # (Nx,   Ny+1) expected
    u_c, v_c = average_to_centers(u2, v2)
    speed = sqrt.(u_c.^2 .+ v_c.^2)   # (Nx, Ny)
    @info @sprintf("speed stats (surface): min=%g max=%g mean=%g", minimum(speed), maximum(speed), mean(speed))
    save_heatmap(speed; lon=lonc, lat=latc, title="Surface speed (m s⁻¹)", fname=joinpath(OUTDIR, "quicklook_speed.png"))
else
    @info "Skipping speed plot (u and/or v not found)."
end
