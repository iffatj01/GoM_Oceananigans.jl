using Oceananigans
using CUDA
using Oceananigans.Units
using Oceananigans.Architectures
using Oceananigans.Utils: TimeInterval, IterationInterval

# ────────────────────────────────────────────────────
# Choose architecture (GPU if available, else CPU)
architecture = CUDA.has_cuda() ? GPU() : CPU()
@info "CUDA is $(CUDA.has_cuda() ? "available" : "NOT available"). Using $(typeof(architecture)) architecture."

# ────────────────────────────────────────────────────
# Define grid (example Gulf of Mexico domain)
grid = LatitudeLongitudeGrid(architecture;
    size      = (128, 128, 30),
    longitude = (-98, -80),
    latitude  = (18, 31),
    z         = (-500, 0),
)

# ────────────────────────────────────────────────────
# Set up the model
model = HydrostaticFreeSurfaceModel(
    grid     = grid,
    buoyancy = SeawaterBuoyancy(),
    coriolis = BetaPlane(latitude = 24),
    closure  = AnisotropicMinimumDissipation(),
    tracers  = (:T, :S),
)

# Initial conditions (simple stratification)
set!(model;
     T = (x, y, z) -> 28 + z / 100,
     S = (x, y, z) -> 35)

# ────────────────────────────────────────────────────
# Create simulation (10 minutes for 1 day)
simulation = Simulation(model; Δt = 10minute, stop_time = 1day)

# Progress callback (every 120 iterations ≈ 2 hours for 10-min Δt)
simulation.callbacks[:progress] = Callback(IterationInterval(120)) do sim
    @info " Simulation time: $(time(sim))"
end

# ────────────────────────────────────────────────────
# Attach an output writer, if available.
function try_attach_writers!(sim)
    # Prefer NetCDF if Oceananigans exposes the writer and NetCDF is installed
    if isdefined(Oceananigans.OutputWriters, :NetCDFOutputWriter) &&
       Base.find_package("NetCDF") !== nothing
        @eval using NetCDF
        NetCDFOutputWriter = getproperty(Oceananigans.OutputWriters, :NetCDFOutputWriter)
        sim.output_writers[:fields] = NetCDFOutputWriter(sim.model;
            fields   = (:u, :v, :T, :S),
            schedule = TimeInterval(6hour),
            filename = "gulf_output.nc",
            overwrite_existing = true,
        )
        @info "Using NetCDFOutputWriter → gulf_output.nc"
        return
    end

    # Fallback: JLD2OutputWriter if exposed and JLD2 is installed
    if isdefined(Oceananigans.OutputWriters, :JLD2OutputWriter) &&
       Base.find_package("JLD2") !== nothing
        @eval using JLD2
        JLD2OutputWriter = getproperty(Oceananigans.OutputWriters, :JLD2OutputWriter)
        sim.output_writers[:fields] = JLD2OutputWriter(sim.model;
            fields   = (:u, :v, :T, :S),
            schedule = TimeInterval(6hour),
            filename = "gulf_output.jld2",
            overwrite_existing = true,
        )
        @info " Using JLD2OutputWriter → gulf_output.jld2"
        return
    end

    # Last resort: Checkpointer (writes full model state; needs JLD2)
    if isdefined(Oceananigans.OutputWriters, :Checkpointer) &&
       Base.find_package("JLD2") !== nothing
        @eval using JLD2
        Checkpointer = getproperty(Oceananigans.OutputWriters, :Checkpointer)
        sim.output_writers[:checkpoint] = Checkpointer(sim.model;
            schedule = TimeInterval(6hour),
            prefix   = "gulf_checkpoint"
        )
        @info " Using Checkpointer (JLD2) → gulf_checkpoint*.jld2"
        return
    end

    @warn "No writable output available. Add NetCDF or JLD2 to this project to enable file output."
end

try_attach_writers!(simulation)

# ────────────────────────────────────────────────────
# Run the simulation
run!(simulation)

