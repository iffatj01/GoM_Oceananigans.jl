# runs/03_seasonal_forcing/run.jl
using Oceananigans
using CUDA
using Oceananigans.Units
using Oceananigans.Utils: TimeInterval, IterationInterval
using Oceananigans.OutputWriters: Checkpointer

# --- Architecture ---
arch = CUDA.has_cuda() ? GPU() : CPU()
@info "CUDA is $(CUDA.has_cuda() ? "available" : "not available"). Using $arch."

# --- Grid (same domain/resolution as your smoke tests) ---
grid = LatitudeLongitudeGrid(arch;
    size      = (160, 120, 30),             # adjust if you want
    longitude = (-98, -80),
    latitude  = (18, 31),
    z         = (-500, 0)
)

# --- Model physics ---
buoyancy = SeawaterBuoyancy()
coriolis = BetaPlane(latitude = 24.0)
closure  = AnisotropicMinimumDissipation()
tracers  = (:T, :S)

# --- Monthly “climatology-like” cycles (simple, tunable) ---
# Units: Qnet [W/m^2] positive warms ocean
#        τx, τy [N/m^2] wind stress
const Qnet_monthly = [120, 140, 160, 190, 210, 230, 220, 200, 170, 150, 130, 120]
const τx_monthly   = [0.05, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.04, 0.05, 0.06, 0.06]
const τy_monthly   = [0.00, 0.00, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, -0.01, -0.01, 0.00, 0.00]

@inline function month_index(t)  # t in seconds
    m = Int(floor(mod(t, 365day) / (30day))) + 1
    return m ≤ 12 ? m : 12
end
qnet(t) = Qnet_monthly[month_index(t)]
τx(t)   = τx_monthly[month_index(t)]
τy(t)   = τy_monthly[month_index(t)]

# --- Surface-layer conversion to tendencies ---
const ρ₀  = 1025.0             # kg/m^3
const Cᴘ  = 3990.0             # J/(kg K)
const hml = 20.0               # mixed-layer thickness [m] used to distribute flux
const NTOP = 1                 # apply in the top NTOP cells (1 = only the top cell)

@inline top_mask(i, j, k, grid) = k > grid.Nz - NTOP ? 1.0 : 0.0

# Temperature: dT/dt = Qnet / (ρ Cᴘ hml)
heat_tendency(i, j, k, grid, clock, fields, p) =
    top_mask(i, j, k, grid) * qnet(clock.time) / (ρ₀ * Cᴘ * hml)

# Wind stress → momentum: du/dt = τx / (ρ hml), dv/dt = τy / (ρ hml)
u_tendency(i, j, k, grid, clock, fields, p) =
    top_mask(i, j, k, grid) * τx(clock.time) / (ρ₀ * hml)

v_tendency(i, j, k, grid, clock, fields, p) =
    top_mask(i, j, k, grid) * τy(clock.time) / (ρ₀ * hml)

forcing = (
    U = Forcing(u_tendency),
    V = Forcing(v_tendency),
    T = Forcing(heat_tendency),
    # You can add freshwater later as another Forcing on S
)

model = HydrostaticFreeSurfaceModel(; grid, buoyancy, coriolis, closure, tracers, forcing)

# --- Initial conditions (same as before) ---
set!(model,
     T = (x, y, z) -> 28 + z / 100,   # weak stratification
     S = (x, y, z) -> 35.0)

# --- Simulation control ---
Δt = 20seconds
stop_time = 60day  # run 2 months to see the seasonal cycle start to matter

sim = Simulation(model; Δt, stop_time)

sim.callbacks[:progress] = Callback(IterationInterval(Int(12hour / Δt))) do sim
    @info "t=$(time(sim)/day) days, Δt=$(sim.Δt) s"
end

# --- Output (always checkpoint; fields if JLD2 available) ---
sim.output_writers[:chk] = Checkpointer(model, TimeInterval(1day);
                                        prefix="out/gulf_seasonal",
                                        force=true)

# Try JLD2 fields
try
    import JLD2
    using Oceananigans.OutputWriters: JLD2OutputWriter
    sim.output_writers[:fields] = JLD2OutputWriter(model;
        fields = (:u, :v, :T, :S, :η),
        schedule = TimeInterval(1day),
        filename = "out/gulf_seasonal_fields.jld2",
        overwrite_existing = true
    )
    @info "Field writer (JLD2) enabled → out/gulf_seasonal_fields.jld2"
catch
    @warn "JLD2 not found; skipping field writer (checkpoints still saved)."
end

@info "Starting seasonal-forcing spinup…"
run!(sim)
@info "Done."
