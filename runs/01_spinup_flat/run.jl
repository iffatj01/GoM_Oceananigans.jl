# runs/01_spinup_flat/run.jl
using Oceananigans
using CUDA
using Oceananigans.Units
using Oceananigans.Architectures
using Oceananigans.OutputWriters
using Oceananigans.Utils: IterationInterval, TimeInterval

# --------------------------
# 1) Choose architecture
# --------------------------
arch = CUDA.has_cuda() ? GPU() : CPU()
@info "CUDA is $(CUDA.has_cuda() ? "available" : "NOT available"). Using $(typeof(arch))."

# --------------------------
# 2) Domain & grid
# 1/8° pilot: lon -98:-80 (18°), lat 18:31 (13°)
# 1/8° ≈ 0.125° → Nx≈144, Ny≈104. We’ll pick 160×120×40 for a buffer.
# Flat bottom for Phase 1.
# --------------------------
grid = LatitudeLongitudeGrid(arch;
    size      = (160, 120, 40),
    longitude = (-98.0, -80.0),
    latitude  = ( 18.0,  31.0),
    z         = (-4000.0, 0.0),
    halo      = (4, 4, 4),
    topology  = (Periodic, Bounded, Bounded)
)

# --------------------------
# 3) Model setup
# --------------------------
buoyancy = SeawaterBuoyancy()
coriolis = BetaPlane(latitude = 24.0)
closure  = AnisotropicMinimumDissipation()

model = HydrostaticFreeSurfaceModel(; 
    grid,
    buoyancy,
    coriolis,
    closure,
    tracers = (:T, :S)
)

# --------------------------
# 4) Initial conditions
# simple stratification; salinity nearly uniform
# --------------------------
set!(model;
    T = (x, y, z) -> 26.0 + z / 200.0,   # warmer at surface
    S = (x, y, z) -> 35.0 + z / 2000.0  # very weak haline strat
)

# --------------------------
# 5) Simulation controls
# NOTE: with a free surface, Δt is limited by fast barotropic waves.
# Keep Δt small initially and increase later after checking stability.
# --------------------------
Δt     = 20seconds
stop_t = 1day

simulation = Simulation(model; Δt = Δt, stop_time = stop_t)

# progress every simulated hour
simulation.callbacks[:progress] = Callback(IterationInterval(3600 ÷ Int(round(Δt)))) do sim
    it   = sim.model.clock.iteration
    time = sim.model.clock.time
    @info " iter=$it, t=$(round(time/3600, digits=2)) h, Δt=$(sim.Δt)s"
    nothing
end

# --------------------------
# 6) Output
# - Checkpointer always works (uses JLD2 internally via Oceananigans).
# - Optional: JLD2 field writer if JLD2 is in your environment.
# --------------------------
mkpath(joinpath(@__DIR__, "../../out"))
chk = Checkpointer(model;
    schedule = TimeInterval(6hour),
    prefix   = joinpath(@__DIR__, "../../out/gulf_checkpoint")
)
simulation.output_writers[:checkpointer] = chk
@info " Checkpointer enabled → out/gulf_checkpoint*.jld2"

# Optional JLD2 fields
try
    @eval using JLD2
    simulation.output_writers[:jld2] = JLD2OutputWriter(model;
        fields   = (:u, :v, :T, :S),
        schedule = TimeInterval(6hour),
        filename = joinpath(@__DIR__, "../../out/gulf_fields.jld2"),
        overwrite_existing = true
    )
    @info " JLD2 field writer enabled → out/gulf_fields.jld2"
catch
    @warn "JLD2 package not found. Skipping field writer (checkpoints still saved)."
end

# --------------------------
# 7) Run!
# --------------------------
@info "Starting simulation…"
run!(simulation)
@info "Done."
