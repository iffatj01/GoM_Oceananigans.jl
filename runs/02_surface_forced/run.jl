# runs/02_surface_forced/run.jl
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
# 2) Grid (GoM-ish box)
grid = LatitudeLongitudeGrid(arch;
    size      = (160, 120, 40),
    longitude = (-98.0, -80.0),
    latitude  = ( 18.0,  31.0),
    z         = (-4000.0, 0.0),
    halo      = (4, 4, 4),
    topology  = (Periodic, Bounded, Bounded),
)

# ─────────────────────────────────────────────────────────────
# 3) Physics
buoyancy = SeawaterBuoyancy()
coriolis = BetaPlane(latitude = 24.0)
closure  = AnisotropicMinimumDissipation()

# ─────────────────────────────────────────────────────────────
# 4) Surface wind as a BODY FORCE (GPU-safe for momentum)
const ρ0 = 1025.0
const τx = 0.05      # N m⁻² (eastward wind stress)
const hF = 30.0      # m  (mixing layer thickness for body forcing)

wind_body_force(x, y, z, t) = (z > -hF) ? (τx / (ρ0 * hF)) : 0.0

# ─────────────────────────────────────────────────────────────
# 5) Surface heat as a TRACER FLUX boundary condition (correct API)
const cp = 3994.0
const Q  = 50.0  # W m⁻² downward positive
heat_flux(x, y, t) = Q / (ρ0 * cp)  # K·m·s⁻¹ equivalent for T

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(heat_flux))
# (Alternatively: top = BoundaryCondition(Flux, heat_flux))

# ─────────────────────────────────────────────────────────────
# 6) Model
model = HydrostaticFreeSurfaceModel(;
    grid, buoyancy, coriolis, closure,
    tracers = (:T, :S),
    forcing = (u = Forcing(wind_body_force),),   # no `field=` keyword
    boundary_conditions = (T = T_bcs,),
)

# ─────────────────────────────────────────────────────────────
# 7) Initial conditions
set!(model;
    T = (x, y, z) -> 26.0 + z / 200.0,
    S = (x, y, z) -> 35.0 + z / 2000.0,
)

# ─────────────────────────────────────────────────────────────
# 8) Simulation
Δt     = 20seconds
stop_t = 2days
simulation = Simulation(model; Δt = Δt, stop_time = stop_t)

progress_every = Int(round(1hour / Δt))
simulation.callbacks[:progress] = Callback(IterationInterval(progress_every)) do sim
    t = sim.model.clock.time
    @info "  t=$(round(t/3600, digits=2)) h, Δt=$(sim.Δt)s"
    nothing
end

# ─────────────────────────────────────────────────────────────
# 9) Output: Checkpointer always; JLD2 fields if available
mkpath(joinpath(@__DIR__, "../../out"))

simulation.output_writers[:checkpointer] = Checkpointer(model;
    schedule = TimeInterval(6hour),
    prefix   = joinpath(@__DIR__, "../../out/gulf_checkpoint_forced"),
)
@info " Checkpointer → out/gulf_checkpoint_forced*.jld2"

try
    @eval using JLD2
    simulation.output_writers[:jld2] = JLD2OutputWriter(model;
        fields   = (:u, :v, :T, :S),
        schedule = TimeInterval(6hour),
        filename = joinpath(@__DIR__, "../../out/gulf_fields_forced.jld2"),
        overwrite_existing = true,
    )
    @info " JLD2 field writer → out/gulf_fields_forced.jld2"
catch
    @warn "JLD2 not found; skipping field writer (checkpoints still saved)."
end

# ─────────────────────────────────────────────────────────────
# 10) Run
@info "Starting surface-forced simulation…"
run!(simulation)
@info "Done."
