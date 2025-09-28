#!/usr/bin/env julia
# 06_spinup_ibgrid_seasonal.jl

using Oceananigans
using JLD2
using Printf
using Statistics

# ---------------------------
# Helper: build grid from primitives stored in JLD2
# ---------------------------
function build_grid_from_jld2(path::AbstractString; arch=CPU())
    @info "Loading IB grid: $path"
    JLD2.jldopen(path, "r") do f
        ks = collect(keys(f))
        @info "Keys found in file: $(join(ks, ", "))"

        required = ("Nx","Ny","Nz","lonW","lonE","latS","latN","Hmax")
        missing = [k for k in required if !haskey(f, k)]
        !isempty(missing) && error("Missing keys $(missing) in $path. Available: $(join(ks, ", ")).")

        Nx   = f["Nx"];   Ny   = f["Ny"];   Nz   = f["Nz"]
        lonW = f["lonW"]; lonE = f["lonE"]; latS = f["latS"]; latN = f["latN"]
        Hmax = f["Hmax"]

        grid = LatitudeLongitudeGrid(arch;
            size      = (Nx, Ny, Nz),
            longitude = (lonW, lonE),
            latitude  = (latS, latN),
            z         = (-Hmax, 0.0)
        )

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

const Δt        = 30.0                    # seconds  (safe for external CFL)
const stop_time = 30 * 24 * 60 * 60       # 30 days

# Wind stress (constant)
const ρ₀  = 1025.0                        # kg m⁻3
const τx  = 0.08                          # N m⁻2 eastward
const τy  = 0.00                          # N m⁻2 northward
const τx_over_ρ = τx / ρ₀
const τy_over_ρ = τy / ρ₀

# Coriolis (f-plane ~ 25°N)
const f0 = 6.0e-5                         # s⁻1

# ---------------------------
# Build grid and model
# ---------------------------
grid = build_grid_from_jld2(IBGRID_PATH; arch=CPU())
@info "Grid size: $(size(grid))"

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx_over_ρ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τy_over_ρ))

# Small background viscosity/diffusivity (helps stability)
closure = ScalarDiffusivity(ν = 20.0, κ = 1.0e-5)

model = HydrostaticFreeSurfaceModel(; grid,
    tracers      = (:T, :S),
    buoyancy     = SeawaterBuoyancy(),
    # Use split-explicit to satisfy fast barotropic CFL via substepping
    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.6),
    coriolis     = FPlane(f = f0),
    closure      = closure,
    boundary_conditions = (u = u_bcs, v = v_bcs),
)

# ---------------------------
# Initial conditions
# ---------------------------
set!(model; T=25.0, S=35.0, u=0.0, v=0.0, η=0.0)
compute!(model.tracers.T); compute!(model.tracers.S)
@info "Initial T/S set after init. extrema(T)=$(extrema(model.tracers.T))  extrema(S)=$(extrema(model.tracers.S))"
@info "Wind stress: τx=$(τx) N m⁻², τy=$(τy) N m⁻² (τ/ρ: $(τx_over_ρ), $(τy_over_ρ)), f=$(f0) s⁻¹"
@info "Closure: ν=$(closure.ν) m² s⁻¹, κ=$(closure.κ) m² s⁻¹"

# ---------------------------
# Snapshot + progress callback
# ---------------------------
const SNAP_EVERY_ITERS = Int(round(6*3600 / Δt))   # every 6 hours
const LOG_EVERY_ITERS  = 200

function save_and_log(sim)
    i  = sim.model.clock.iteration
    t  = sim.model.clock.time
    dt = sim.Δt

    if i % LOG_EVERY_ITERS == 0
        @info @sprintf("iter=%7d  t=%10.1f s (%.2f d)  Δt=%.1f s", i, t, t/86400, dt)
    end

    if i % SNAP_EVERY_ITERS == 0
        T = sim.model.tracers.T
        S = sim.model.tracers.S
        η = sim.model.free_surface.η
        U = sim.model.velocities.u
        V = sim.model.velocities.v
        compute!(T); compute!(S); compute!(η); compute!(U); compute!(V)

        Tcpu = Array(interior(T))
        Scpu = Array(interior(S))
        ηcpu = Array(interior(η))
        Ucpu = Array(interior(U))
        Vcpu = Array(interior(V))

        fpath = joinpath(OUT_DIR, @sprintf("snap_%07d.jld2", i))
        JLD2.jldsave(fpath; T=Tcpu, S=Scpu, η=ηcpu, u=Ucpu, v=Vcpu,
                     iteration=i, time_seconds=t, grid_size=size(sim.model.grid))

        @info "Saved snapshot: $(basename(fpath))  extrema(T)=$(extrema(T))  |u|∈($(minimum(abs.(U))), $(maximum(abs.(U))))"
    end
    return nothing
end

# ---------------------------
# Simulation
# ---------------------------
sim = Simulation(model; Δt=Δt, stop_time=stop_time)
add_callback!(sim, save_and_log, IterationInterval(1))

@info "Starting run… grid=$(size(grid))  Δt=$Δt  stop=$(Float64(stop_time))"
run!(sim)
@info "Done. Outputs at: $OUT_DIR"
