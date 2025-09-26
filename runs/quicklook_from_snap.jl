# runs/quicklook_from_snap.jl
# Usage:
#   julia --project=. runs/quicklook_from_snap.jl [optional_path_to_snap.jld2]

using JLD2
using Printf
using Statistics
using Plots

default(fmt = :png)

# ------------------------------------------------------------
# Find newest snap in outputs/ if no path is provided
# ------------------------------------------------------------
function newest_snap_in(dir::AbstractString)
    files = filter(f -> occursin(r"^snap_\d{7}\.jld2$", f), readdir(dir))
    isempty(files) && error("No snap_XXXXXXX.jld2 files found in $(dir).")
    joinpath(dir, sort(files)[end])
end

# ------------------------------------------------------------
# List top-level keys in a JLD2 file using only stable API
# ------------------------------------------------------------
function top_keys(path::AbstractString)
    JLD2.jldopen(path, "r") do f
        return collect(String.(keys(f)))
    end
end

# ------------------------------------------------------------
# Read a dataset if it exists; return `nothing` if not.
# Uses only haskey and read.
# ------------------------------------------------------------
function read_if_exists(path::AbstractString, name::AbstractString)
    JLD2.jldopen(path, "r") do f
        haskey(f, name) || return nothing
        return read(f, name)   # returns the dataset value
    end
end

# ------------------------------------------------------------
# Return a 2D array from either a 2D field or the surface of a 3D field
# ------------------------------------------------------------
function as_2d_surface(A)
    A === nothing && return nothing
    nd = ndims(A)
    if nd == 2
        return Array(A)
    elseif nd == 3
        return Array(@view A[:, :, end])  # surface slice
    else
        return nothing
    end
end

# ------------------------------------------------------------
# Reconstruct lon/lat from ibgrid JLD2 using only haskey/read
# ------------------------------------------------------------
function reconstruct_lonlat_from_ibgrid()
    ibfile = joinpath(@__DIR__, "..", "data", "processed", "ibgrid_gom.jld2")
    isfile(ibfile) || return nothing, nothing

    lonW = lonE = latS = latN = Nx = Ny = nothing
    JLD2.jldopen(ibfile, "r") do f
        haskey(f, "lonW") && (lonW = read(f, "lonW"))
        haskey(f, "lonE") && (lonE = read(f, "lonE"))
        haskey(f, "latS") && (latS = read(f, "latS"))
        haskey(f, "latN") && (latN = read(f, "latN"))
        haskey(f, "Nx")   && (Nx   = read(f, "Nx"))
        haskey(f, "Ny")   && (Ny   = read(f, "Ny"))
    end

    if any(x -> x === nothing, (lonW, lonE, latS, latN, Nx, Ny))
        return nothing, nothing
    end

    lon = collect(range(float(lonW), float(lonE), length = Int(Nx)))
    lat = collect(range(float(latS), float(latN), length = Int(Ny)))
    return lon, lat
end

# ------------------------------------------------------------
# Save a heatmap with safe clims (handles constant fields)
# ------------------------------------------------------------
function save_heatmap(data; lon=nothing, lat=nothing, title="", fname="")
    A = Array(data)
    amin, amax = extrema(A)
    clims = amax == amin ? (amin - 1e-6, amax + 1e-6) : (amin, amax)

    if lon !== nothing && lat !== nothing &&
       length(lon) == size(A, 1) && length(lat) == size(A, 2)
        p = heatmap(lon, lat, A'; aspect_ratio=:equal,
                    xlabel="lon", ylabel="lat", title=title,
                    clims=clims, colorbar=true)
    else
        p = heatmap(A'; aspect_ratio=:equal,
                    title=title, clims=clims, colorbar=true)
    end
    savefig(p, fname)
    return nothing
end

# ----------------- main -----------------
SNAP = length(ARGS) >= 1 ? ARGS[1] : newest_snap_in(joinpath(@__DIR__, "..", "outputs"))
@info "Reading snapshot" file=SNAP

# Show keys so you can see what's in the file
keys_in_file = try
    top_keys(SNAP)
catch err
    @warn "Could not list top-level keys: $err"
    String[]
end
!isempty(keys_in_file) && @info "Top-level keys in file: $(join(keys_in_file, ", "))"

# Load fields if present (names must match what your writer saved)
T  = read_if_exists(SNAP, "T")
η  = read_if_exists(SNAP, "η")   # sometimes already 2D

# Optional lon/lat axes
lon, lat = reconstruct_lonlat_from_ibgrid()

# Output directory for images (same as snapshot dir)
outdir = dirname(SNAP)

# Temperature quicklook (surface if 3D)
T2 = as_2d_surface(T)
if T2 !== nothing
    save_heatmap(T2; lon, lat,
        title = @sprintf("Surface Temperature (°C)  extrema=(%.3g, %.3g)", minimum(T2), maximum(T2)),
        fname = joinpath(outdir, "quicklook_T.png"))
    @info "Saved: $(joinpath(outdir, "quicklook_T.png"))  (extrema = $(extrema(T2)))"
else
    @info "Skipping T plot (temperature not found or unsupported shape)."
end

# Free surface quicklook (η is often 2D; if 3D, use surface)
eta2 = as_2d_surface(η)
if eta2 !== nothing
    save_heatmap(eta2; lon, lat,
        title = @sprintf("Free surface η (m)  extrema=(%.3g, %.3g)", minimum(eta2), maximum(eta2)),
        fname = joinpath(outdir, "quicklook_eta.png"))
    @info "Saved: $(joinpath(outdir, "quicklook_eta.png"))  (extrema = $(extrema(eta2)))"
else
    @info "Skipping η plot (η not found or unsupported shape)."
end

