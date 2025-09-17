# runs/05_surface_seasonal.jl

using Oceananigans
using CUDA
using Oceananigans.Units
using Oceananigans.Architectures
using Oceananigans.OutputWriters
using Oceananigans.Utils: IterationInterval, TimeInterval
using Oceananigans.BoundaryConditions

# ─────────────────────────────────────────────────────────────
# 1) Architecture
arch = CUDA.has_cuda() ? GPU() : CPU()
@info "CUDA is $(CUDA.has_cuda() ? "available" : "NOT available"). Using $(typeof(arch))."

# ─────────────────────────────────────────────────────────────
# 2) Grid
grid = LatitudeLongitudeGrid(arch;
    size      = (160, 120, 40),
    longitude = (-98.0, -80.0),
    latitude  = ( 18.0,  31.0),
    z         = (-4000.0, 0.0),
    halo      = (4, 4, 4),
    topology  = (Bounded, Bounded, Bounded),
)

# ─────────────────────────────────────────────────────────────
# 3) Physics
buoyancy = SeawaterBuoyancy()
coriolis = BetaPlane(latitude = 24.0)
closure  = AnisotropicMinimumDissipation()

# ─────────────────────────────────────────────────────────────
# 4) Seasonal surface forcing (GPU-safe)
const ρ0   = 1025.0
const cp   = 3994.0
const hF   = 30.0                  # wind body-force layer thickness
const YEAR = 365.0 * 86400.0       # seconds in a year (Float64 literal)

const τ0   = 0.03      # N m⁻² mean
const τamp = 0.02      # N m⁻² amplitude
const Q0   =  75.0     # W m⁻² mean (downward = heating)
const Qamp =  50.0     # W m⁻² amplitude

@inline τx_of_t(t::Float64) = τamp == 0 ? τ0 : (τ0 + τamp * cos(2π * t / YEAR))
@inline Q_of_t(t::Float64)  = Qamp == 0 ? Q0 : (Q0 + Qamp * cos(2π * (t / YEAR - 0.25)))

# Wind stress as body force in upper hF
@inline function wind_body_force(x, y, z, t)
    return z > -hF ? τx_of_t(t) / (ρ0 * hF) : 0.0
end

# IMPORTANT: positive flux cools, so apply minus to heat when Q>0 downward
@inline function heat_flux(x, y, t)
    return - Q_of_t(t) / (ρ0 * cp)
end

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(heat_flux))

# ─────────────────────────────────────────────────────────────
# 5) Model
model = HydrostaticFreeSurfaceModel(;
    grid, buoyancy, coriolis, closure,
    tracers = (:T, :S),
    forcing = (u = Forcing(wind_body_force),),
    boundary_conditions = (T = T_bcs,),
)

# ─────────────────────────────────────────────────────────────
# 6) Initial conditions
set!(model;
    T = (x, y, z) -> 26.0 + z / 200.0,
    S = (x, y, z) -> 35.0 + z / 2000.0,
)

# ─────────────────────────────────────────────────────────────
# 7) Simulation
Δt       = 20seconds
stop_t   = 30days
simulation = Simulation(model; Δt = Δt, stop_time = stop_t)

progress_every = Int(round(1hour / Δt))
simulation.callbacks[:progress] = Callback(IterationInterval(progress_every)) do sim
    t = sim.model.clock.time
    @info "  t=$(round(t/86400, digits=3)) days, Δt=$(sim.Δt)s"
    nothing
end

# ─────────────────────────────────────────────────────────────
# 8) Output
outdir = normpath(joinpath(@__DIR__, "..", "out"))
mkpath(outdir)

simulation.output_writers[:checkpointer] = Checkpointer(model;
    schedule = TimeInterval(5days),
    prefix   = joinpath(outdir, "gulf_seasonal_checkpoint"),
)
@info "Checkpointer → $(outdir)/gulf_seasonal_checkpoint*.jld2"

try
    @eval using JLD2
    simulation.output_writers[:jld2] = JLD2OutputWriter(model;
        fields   = (:u, :v, :T, :S),
        schedule = TimeInterval(1day),
        filename = joinpath(outdir, "gulf_seasonal_fields.jld2"),
        overwrite_existing = true,
    )
    @info "Field writer  → $(outdir)/gulf_seasonal_fields.jld2 (daily)"
catch
    @warn "JLD2 not found; skipping field writer (checkpoints still saved)."
end

# ─────────────────────────────────────────────────────────────
# 9) Run
@info "Starting seasonal surface-forced run…"
run!(simulation)
@info "Done."

