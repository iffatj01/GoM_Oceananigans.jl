
# preprocess/make_ibgrid_from_bathy.jl
#
# Build an ImmersedBoundaryGrid from a local bathymetry cutout
# saved at data/raw/bathymetry.nc (lat × lon, variable "z" as depth +down).

using Oceananigans
using Oceananigans.Architectures: CPU, on_architecture
using NCDatasets
using JLD2
using Printf
using Interpolations: interpolate, extrapolate, Gridded, Linear, Flat

# -------------------------
# Domain box (match your runs)
# -------------------------
const Nx, Ny, Nz = 160, 120, 40
const lonW, lonE = -98.0, -80.0
const latS, latN =  18.0,  31.0
const Hmax       = 4000.0

# -------------------------
# Paths
# -------------------------
const raw_nc  = joinpath(@__DIR__, "..", "data", "raw", "bathymetry.nc")
const proc_jl = joinpath(@__DIR__, "..", "data", "processed", "ibgrid_gom.jld2")

# -------------------------
# Build and save IB grid from a bathy function h(lon, lat)
# -------------------------
function build_and_save_ibgrid(hfun)
    # IMPORTANT: build on CPU so the saved file is portable
    grid = LatitudeLongitudeGrid(CPU();
        size      = (Nx, Ny, Nz),
        longitude = (lonW, lonE),
        latitude  = (latS, latN),
        z         = (-Hmax, 0.0),
        topology  = (Bounded, Bounded, Bounded),
    )

    ibgrid = ImmersedBoundaryGrid(grid, GridFittedBottom(hfun))

    # Ensure we truly save a CPU object (even if someone changes the code later)
    ibgrid_cpu = on_architecture(CPU(), ibgrid)

    mkpath(dirname(proc_jl))
    @save proc_jl ibgrid ibgrid_cpu Hmax lonW lonE latS latN Nx Ny Nz
    @info "Saved processed immersed-boundary grid → $(proc_jl)"
end

# -------------------------
# Load local bathy and construct h(lon, lat)
# -------------------------
@info "Building immersed-boundary grid from $(raw_nc)…"

if !isfile(raw_nc)
    error("Bathymetry file not found: $(raw_nc). Run preprocess/download_bathy_cutout.jl first.")
end

ds  = NCDataset(raw_nc)
lon = haskey(ds, "lon") ? vec(ds["lon"][:]) :
      haskey(ds, "longitude") ? vec(ds["longitude"][:]) :
      error("No lon/longitude variable found in $(raw_nc).")

lat = haskey(ds, "lat") ? vec(ds["lat"][:]) :
      haskey(ds, "latitude") ? vec(ds["latitude"][:]) :
      error("No lat/latitude variable found in $(raw_nc).")

Z   = haskey(ds, "z") ? Array(ds["z"][:,:]) :
      haskey(ds, "elevation") ? -Array(ds["elevation"][:,:]) :
      error("No z/elevation variable found in $(raw_nc).")

close(ds)

# Ensure Z matches coordinate order; prefer (lat, lon)
Zlatlon = size(Z) == (length(lat), length(lon))  ? Z :
          size(Z) == (length(lon), length(lat)) ? permutedims(Z, (2, 1)) :
          error(@sprintf("Unexpected Z size %s for lat=%d, lon=%d", string(size(Z)), length(lat), length(lon)))

# Clamp depths to [5, Hmax] (positive-down depths expected)
Zlatlon = clamp.(Zlatlon, 5.0, Hmax)

# Build bilinear interpolation in (lat, lon) with flat extrapolation
itp = extrapolate(interpolate((lat, lon), Float32.(Zlatlon), Gridded(Linear())), Flat())

# Define h(lon, lat) as positive-down depth
h(lon_val, lat_val) = clamp(Float64(itp(lat_val, lon_val)), 5.0, Hmax)

build_and_save_ibgrid(h)
