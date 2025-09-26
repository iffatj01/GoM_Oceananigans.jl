#!/usr/bin/env julia
# 06_spinup_ibgrid_seasonal.jl

using Oceananigans
using JLD2
using Printf
using Statistics

# ---------------------------
# Helper: load primitive params and build a fresh grid
# ---------------------------
function build_grid_from_jld2(path::AbstractString; arch=CPU())
    @info "Loading IB grid: $path"
    JLD2.jldopen(path, "r") do f
        ks = collect(keys(f))
        @info "Keys found in file: $(join(ks, ", "))"

        # Ensure required keys exist in the file
        required = ("Nx","Ny","Nz","lonW","lonE","latS","latN","Hmax")
        missing = [k for k in required if !haskey(f, k)]
        if !isempty(missing)
            error("Missing keys $(missing) in $path. Available keys: $(join(ks, ", ")).")
        end

        Nx   = f["Nx"];   Ny   = f["Ny"];   Nz   = f["Nz"]
        lonW = f["lonW"]; lonE = f["lonE"]; latS = f["latS"]; latN = f["latN"]
        Hmax = f["Hmax"]

        grid = LatitudeLongitudeGrid(arch;
            size      = (Nx, Ny, Nz),
            longitude = (lonW, lonE),
            latitude  = (latS, latN),
            z         = (-Hmax, 0.0)
        )

        # Ignore serialized grid objects on disk (avoid type reconstruction issues)
        if ("ibgrid" in ks) || ("grid" in ks)
            @warn "Serialized grid objects found in file; ignoring them and rebuilding the grid from primitives."
        end

        return grid
    end
end

# ---------------------------
# Paths and run parameters
# ---------------------------
const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR    = joinpath(PROJECT_DIR, "data", "processed")
const OUT_DIR     = joinpath(PROJECT_DIR, "outputs")
mkpath(OUT_DIR)

const IBGRID_PATH = joinpath(DATA_DIR, "ibgrid_gom.jld2")

const Δt        = 300.0              # seconds
const stop_time = 5 * 24 * 60 * 60   # 5 days in seconds

# ---------------------------
# Build grid and model
# ---------------------------
grid = build_grid_from_jld2(IBGRID_PATH; arch=CPU())
@info "Grid size: $(size(grid))"

model = HydrostaticFreeSurfaceModel(; grid,
    tracers      = (:T, :S),
    buoyancy     = SeawaterBuoyancy(),
    free_surface = ExplicitFreeSurface(),
)

# ---------------------------
# Initial conditions
# ---------------------------
set!(model; T=25.0, S=35.0, u=0.0, v=0.0, η=0.0)
compute!(model.tracers.T); compute!(model.tracers.S)
@info "Initial T/S set after init. extrema(T)=$(extrema(model.tracers.T))  extrema(S)=$(extrema(model.tracers.S))"

# ---------------------------
# Snapshot + progress callback (manual saver, no OutputWriter)
# ---------------------------
const SNAP_EVERY_ITERS = 72   # every 6 hours for Δt=300s
const LOG_EVERY_ITERS  = 50

function save_and_log(sim)
    i  = sim.model.clock.iteration
    t  = sim.model.clock.time
    dt = sim.Δt

    if i % LOG_EVERY_ITERS == 0
        @info @sprintf("iter=%6d  t=%9.1f s (%.2f d)  Δt=%.1f s", i, t, t/86400, dt)
    end

    if i % SNAP_EVERY_ITERS == 0
        T = sim.model.tracers.T
        S = sim.model.tracers.S
        η = sim.model.free_surface.η
        compute!(T); compute!(S); compute!(η)

        # move to CPU arrays for saving
        Tcpu = Array(interior(T))
        Scpu = Array(interior(S))
        ηcpu = Array(interior(η))

        fpath = joinpath(OUT_DIR, @sprintf("snap_%07d.jld2", i))
        JLD2.jldsave(fpath; T=Tcpu, S=Scpu, η=ηcpu,
                     iteration=i, time_seconds=t,
                     grid_size=size(sim.model.grid))

        @info "Saved snapshot: $(basename(fpath))   extrema(T)=$(extrema(T))"
    end
    return nothing
end

# ---------------------------
# Simulation
# ---------------------------
# NOTE: remove unsupported `iteration_interval` keyword
sim = Simulation(model; Δt=Δt, stop_time=stop_time)

# Drive our own cadence in the callback (runs every iteration; we gate inside)
add_callback!(sim, save_and_log, IterationInterval(1))

@info "Starting run… grid=$(size(grid))  Δt=$Δt  stop=$(Float64(stop_time))"
run!(sim)
@info "Done. Outputs at: $OUT_DIR"
