


# runs/04_bathymetry_analytic.jl

using Oceananigans
using Oceananigans.Units: hour, day
using Oceananigans.Utils: TimeInterval, IterationInterval  # ← needed for schedules

# -------------------------
# Domain and grid
# -------------------------
const Nx, Ny, Nz = 160, 120, 40
const lonW, lonE = -98.0, -80.0
const latS, latN =  18.0,  31.0
const Hmax       = 3500.0

grid = LatitudeLongitudeGrid(CPU();
    size      = (Nx, Ny, Nz),
    longitude = (lonW, lonE),
    latitude  = (latS, latN),
    z         = (-Hmax, 0.0),
    topology  = (Bounded, Bounded, Bounded),
)

@info "Grid built" Nx Ny Nz lonW lonE latS latN Hmax

# -------------------------
# Analytic bathymetry h(λ, φ)
# -------------------------
λ0 = (lonW + lonE) / 2
φ0 = (latS + latN) / 2

h(λ, φ) = begin
    raw = Hmax * (0.45 + 0.55 * exp(-((λ - λ0)^2 + (φ - φ0)^2) / 100))
    clamp(raw, 5.0, Hmax)
end

# Grid-fitted immersed bottom
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

# -------------------------
# Physics / model
# -------------------------
eos = LinearEquationOfState(;
    thermal_expansion   = 2e-4,   # 1/K
    haline_contraction  = 7.5e-4  # 1/psu
)

buoyancy     = SeawaterBuoyancy(equation_of_state = eos)
coriolis     = BetaPlane(latitude = (latS + latN) / 2)
closure      = AnisotropicMinimumDissipation()
free_surface = SplitExplicitFreeSurface(substeps = 21)

model = HydrostaticFreeSurfaceModel(
    grid         = ib_grid,
    tracers      = (:T, :S),
    buoyancy     = buoyancy,
    coriolis     = coriolis,
    closure      = closure,
    free_surface = free_surface,
    timestepper  = :QuasiAdamsBashforth2,
)

# -------------------------
# Initial conditions
# -------------------------
set!(model, u = 0.0, v = 0.0, T = 20.0, S = 35.0, η = 0.0)

# -------------------------
# Simulation
# -------------------------
Δt        = 300.0      # 5 minutes
stop_time = 1day

sim = Simulation(model, Δt = Δt, stop_time = stop_time)

# Print progress every simulated hour (FIX: pass a schedule, not `every=`)
sim.callbacks[:progress] = Callback(TimeInterval(1hour)) do s
    t_hours = round(s.model.clock.time / 3600; digits = 2)
    @info "t = $(t_hours) h, Δt = $(s.Δt) s"
    nothing
end

run!(sim)
