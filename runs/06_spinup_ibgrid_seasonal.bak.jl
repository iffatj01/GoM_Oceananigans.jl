# runs/06_spinup_ibgrid_seasonal.jl
#
# Robust spinup on an ImmersedBoundaryGrid with seasonal-forcing placeholders.
# - Safely loads the IB grid from a JLD2 file regardless of the variable name.
# - Avoids pickup unless checkpoint files actually exist.

using Oceananigans
using Oceananigans.Simulations: Simulation, run!
using Oceananigans.OutputWriters
using Oceananigans.Units: minute, hour, day
using JLD2

# -----------------------
# Paths and configuration
# -----------------------
const PROJROOT   = dirname(dirname(@__FILE__))             # repo root
const DATADIR    = joinpath(PROJROOT, "data", "processed")
const OUTDIR     = joinpath(PROJROOT, "outputs")
const IBGRIDFILE = joinpath(DATADIR, "ibgrid_gom.jld2")

isdir(OUTDIR) || mkpath(OUTDIR)

# Timestep and run length
const Δt       = 5minute
const SPINUP_D = 5day

# Uniform initial conditions (replace with your fields if desired)
const T0 = 25.0
const S0 = 35.0

# ----------------
# Helpers
# ----------------

"""
    load_ibgrid(path) -> grid

Open `path` (JLD2) and return the first value that looks like an
Oceananigans grid (e.g., `ImmersedBoundaryGrid` or any `AbstractGrid`).
If no such value exists, throw a clear error listing the available keys.
"""
function load_ibgrid(path::AbstractString)
    @assert isfile(path) "IB grid file not found at $path"

    grid_like_types = (
        Oceananigans.Grids.AbstractGrid,
        Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid,
    )

    JLD2.jldopen(path, "r") do f
        ks = collect(keys(f))
        # Try common names first
        common_names = ("grid", "ibgrid", "immersed_boundary_grid", "IBGrid", "domain_grid")

        for name in common_names
            if name in ks
                v = read(f, name)
                for T in grid_like_types
                    if v isa T
                        @info "Loaded grid from key $(name)" size = size(v)
                        return v
                    end
                end
            end
        end

        # Fallback: scan all top-level keys and pick the first grid-like object
        for k in ks
            v = read(f, k)
            for T in grid_like_types
                if v isa T
                    @info "Loaded grid from key $(k)" size = size(v)
                    return v
                end
            end
        end

        # Clear error with available keys listed
        msg = "No grid-like object found in $path. " *
              "Available keys: $(join(ks, \", \")). " *
              "Open the file (e.g. JLD2.jldopen(path, \"r\") do f; keys(f); end) to inspect its contents."
        error(msg)
    end
end

# Only pass pickup=true if checkpoint files exist
function safe_run!(sim::Simulation; pickup::Bool=false)
    if pickup
        have_chk = any(startswith(fn, "chk_") && endswith(fn, ".jld2") for fn in readdir(OUTDIR))
        if have_chk
            return run!(sim; pickup=true)
        else
            @warn "Pickup requested but no checkpoints found; running from t=0 instead."
        end
    end
    return run!(sim)
end

# ----------------
# Load IB grid
# ----------------
@info "Loading IB grid" ibgrid_file = IBGRIDFILE
ibgrid = load_ibgrid(IBGRIDFILE)

# ----------------
# Build the model
# ----------------
@info "Building HydrostaticFreeSurfaceModel (IB on CPU)"
model = HydrostaticFreeSurfaceModel(
    grid         = ibgrid,
    tracers      = (:T, :S),
    free_surface = ExplicitFreeSurface()
)
@info "Built model via: HFS with momentum_advection & tracer_advection & free_surface"

# ----------------------------
# Simulation & output writers
# ----------------------------
simulation = Simulation(model, Δt = Δt)

# Snapshots every 6 hours
simulation.output_writers[:snapshots] = JLD2OutputWriter(model;
    fields   = (; T = model.tracers.T, S = model.tracers.S, η = model.free_surface.η),
    schedule = TimeInterval(6hour),
    dir      = OUTDIR,
    prefix   = "snap",
    force    = true,
)

# Checkpointer every 6 hours (for future restarts)
simulation.output_writers[:checkpointer] = Checkpointer(model;
    schedule = TimeInterval(6hour),
    dir      = OUTDIR,
    prefix   = "chk",
)

# ----------------------------
# Priming run (initialization)
# ----------------------------
@info "Priming model (initialization-only run)…"
simulation.stop_time = 0.0
safe_run!(simulation)

# ------------------------------------
# Apply ICs *after* the priming run
# ------------------------------------
set!(model, T = T0, S = S0)
@info "Initial T/S set after init" extrema_T = (T0, T0) extrema_S = (S0, S0)

# ----------------------------
# Main spinup integration
# ----------------------------
simulation.stop_time = SPINUP_D
@info "Starting run…" grid = size(ibgrid) Δt = Δt stop = simulation.stop_time
safe_run!(simulation; pickup=false)   # first run: no pickup

@info "Done. Outputs at: $OUTDIR"

