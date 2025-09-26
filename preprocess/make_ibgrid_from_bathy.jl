using Oceananigans
using JLD2, Printf

# ---- domain settings (adjust if you like) ----
Nx, Ny, Nz = 160, 120, 40
lonW, lonE = -97.0, -80.0
latS, latN =  18.0,  31.5
Hmax       = 4000.0              # deepest point (m, positive depth)

# ---- base grid: z from -Hmax to 0 (top at 0) ----
grid = LatitudeLongitudeGrid(;
    topology  = (Bounded, Bounded, Bounded),
    size      = (Nx, Ny, Nz),
    longitude = (lonW, lonE),
    latitude  = (latS, latN),
    z         = (-Hmax, 0.0)
)

# ---- BATHYMETRY: synthetic, strictly 0 ≤ H(λ,φ) ≤ Hmax ----
H2D(λ, φ) = 0.6Hmax * (0.55 + 0.45 * cospi((λ - lonW) / (lonE - lonW))
                                 * sinpi((φ - latS) / (latN - latS)))
Hfield = Field{Center, Center, Nothing}(grid)
set!(Hfield, H2D)

# ---- Build immersed boundary grid ----
bottom = GridFittedBottom(Hfield)            # H is depth (m, positive), bottom at z = -H
ibgrid = ImmersedBoundaryGrid(grid, bottom)

# ---- Save exactly what the spinup expects ----
out = joinpath(@__DIR__, "..", "data", "processed"); mkpath(out)
outfile = joinpath(out, "ibgrid_gom.jld2")
@save outfile ibgrid Hmax lonW lonE latS latN Nx Ny Nz
@info "Saved IB grid" file=outfile
