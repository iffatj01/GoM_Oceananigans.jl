# runs/04_bathymetry_analytic/run.jl

using Oceananigans
using CUDA
using Oceananigans.Units
using Oceananigans.Utils: TimeInterval, IterationInterval
using Oceananigans.OutputWriters: Checkpointer

arch = CUDA.has_cuda() ? GPU() : CPU()
@info "Using $arch"

# Same horizontal domain, deeper vertical to accommodate slopes
grid = LatitudeLongitudeGrid(arch;
    size      = (160, 120, 40),
    longitude = (-98, -80),
    latitude  = (18, 31),
    z         = (-4000, 0)
)

# "Pseudo-bathymetry" mask via a damping (sponge) near coasts to emulate shelves.
# This keeps the flow off very shallow regions before we wire true bathymetry.
const sponge_width = 8           # cells from each landward side
const λ_sponge = 1 / (6hour)     # damping rate

# Damps momentum in a rim around the domain (simple/robust)
function sponge_mask(i, j, k, grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    m = 0.0
    if i ≤ sponge_width || i ≥ Nx - sponge_width + 1 || j ≤ sponge_width || j ≥ Ny - sponge_width + 1
        m = 1.0
    end
    return m
end

du_sponge(i, j, k, grid, clock, fields, p) = -λ_sponge * sponge_mask(i, j, k, grid) * (@inbounds fields.u[i, j, k])
dv_sponge(i, j, k, grid, clock, fields, p) = -λ_sponge * sponge_mask(i, j, k, grid) * (@inbounds fields.v[i, j, k])

buoyancy = SeawaterBuoyancy()
coriolis = BetaPlane(latitude = 24.0)
closure  = AnisotropicMinimumDissipation()
tracers  = (:T, :S)

forcing = (U = Forcing(du_sponge), V = Forcing(dv_sponge))

model = HydrostaticFreeSurfaceModel(; grid, buoyancy, coriolis, closure, tracers, forcing)

set!(model, T = (x, y, z) -> 28 + z / 100,
            S = (x, y, z) -> 35.0)

Δt = 20seconds
stop_time = 10day

sim = Simulation(model; Δt, stop_time)
sim.callbacks[:progress] = Callback(IterationInterval(Int(12hour / Δt))) do sim
    @info "t=$(time(sim)/day) days"
end

sim.output_writers[:chk] = Checkpointer(model, TimeInterval(1day);
                                        prefix="out/gulf_analytic_bathy",
                                        force=true)

@info "Starting analytic-bathymetry test…"
run!(sim)
@info "Done."
