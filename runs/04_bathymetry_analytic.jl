#!/usr/bin/env julia

using Oceananigans
using Oceananigans.Units: hours, days
using Oceananigans.OutputWriters: Checkpointer, TimeInterval
using Printf

# ------------------------------
# Domain, resolution, bathymetry
# ------------------------------
λW, λE = -98.0, -80.0           # °E (negative = °W)
φS, φN =  18.0,  31.0           # °N
Nx, Ny, Nz = 160, 120, 40
Hmax = 3500.0                   # m

# Gulf-like bowl with shelf (analytic)
function depth(λ, φ)
    λc, φc = -89.5, 24.5
    r = sqrt((λ - λc)^2 + (φ - φc)^2)
    basin = Hmax
    shelf = 150.0
    d = clamp((r - 4.0) / 6.0, 0.0, 1.0)   # shelf→basin transition
    return shelf * (1 - d) + basin * d
end
h(λ, φ) = -depth(λ, φ)  # negative below sea level

# ------------------------------
# Build grid with bathymetry
# ------------------------------
function build_grid()
    # Prefer immersed_boundary keyword if available, else wrapper grid
    try
        IB = Oceananigans.ImmersedBoundaries
        return Oceananigans.Grids.LatitudeLongitudeGrid(
            size = (Nx, Ny, Nz),
            longitude = (λW, λE),
            latitude  = (φS, φN),
            z = (-Hmax, 0.0),
            topology = (Bounded, Bounded, Bounded),
            immersed_boundary = IB.ImmersedBoundary(IB.Topography(h)),
        )
    catch
        try
            IB = Oceananigans.ImmersedBoundaries
            base = Oceananigans.Grids.LatitudeLongitudeGrid(
                size = (Nx, Ny, Nz),
                longitude = (λW, λE),
                latitude  = (φS, φN),
                z = (-Hmax, 0.0),
                topology = (Bounded, Bounded, Bounded),
            )
            return IB.ImmersedBoundaryGrid(base, IB.GridFittedBottom(h))
        catch
            @warn "Immersed-boundary topography not available. Using flat bottom."
            return Oceananigans.Grids.LatitudeLongitudeGrid(
                size = (Nx, Ny, Nz),
                longitude = (λW, λE),
                latitude  = (φS, φN),
                z = (-Hmax, 0.0),
                topology = (Bounded, Bounded, Bounded),
            )
        end
    end
end

grid = build_grid()
@info "Grid built" Nx Ny Nz λW λE φS φN Hmax

# ------------------------------
# Physics: coriolis, EOS, closure
# ------------------------------
function make_coriolis()
    try
        return BetaPlane(latitude = 25.0)
    catch
        Ω = 7.292115e-5
        φ = 25.0 * π/180
        R = 6.371e6
        f0 = 2Ω * sin(φ)
        β  = 2Ω * cos(φ) / R
        return BetaPlane(f₀ = f0, β = β)
    end
end
βplane = make_coriolis()

# EOS API for your version: use thermal_expansion / haline_contraction
eos      = LinearEquationOfState(thermal_expansion = 2e-4,
                                 haline_contraction = 7.4e-4)

# SeawaterBuoyancy: DO NOT pass reference_density on this version
buoyancy = SeawaterBuoyancy(equation_of_state = eos)

closure  = AnisotropicMinimumDissipation()
free_surface = SplitExplicitFreeSurface()

# ------------------------------
# Lateral BCs (simple, stable) + tracers
# ------------------------------
BC  = Oceananigans.BoundaryConditions

# Zero-flux (free-slip/zero-gradient) everywhere for now
u_bcs = BC.FieldBoundaryConditions(
    west   = BC.FluxBoundaryCondition(0.0),
    east   = BC.FluxBoundaryCondition(0.0),
    south  = BC.FluxBoundaryCondition(0.0),
    north  = BC.FluxBoundaryCondition(0.0),
    bottom = BC.FluxBoundaryCondition(0.0),
    top    = BC.FluxBoundaryCondition(0.0),
)
v_bcs = BC.FieldBoundaryConditions(
    west   = BC.FluxBoundaryCondition(0.0),
    east   = BC.FluxBoundaryCondition(0.0),
    south  = BC.FluxBoundaryCondition(0.0),
    north  = BC.FluxBoundaryCondition(0.0),
    bottom = BC.FluxBoundaryCondition(0.0),
    top    = BC.FluxBoundaryCondition(0.0),
)
T_bcs = BC.FieldBoundaryConditions(
    west = BC.FluxBoundaryCondition(0.0), east = BC.FluxBoundaryCondition(0.0),
    south = BC.FluxBoundaryCondition(0.0), north = BC.FluxBoundaryCondition(0.0),
    top = BC.FluxBoundaryCondition(0.0), bottom = BC.FluxBoundaryCondition(0.0),
)
S_bcs = BC.FieldBoundaryConditions(
    west = BC.FluxBoundaryCondition(0.0), east = BC.FluxBoundaryCondition(0.0),
    south = BC.FluxBoundaryCondition(0.0), north = BC.FluxBoundaryCondition(0.0),
    top = BC.FluxBoundaryCondition(0.0), bottom = BC.FluxBoundaryCondition(0.0),
)
bcs = (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)

# ------------------------------
# Weak sponge near walls
# ------------------------------
σ₀ = 1.0 / (3days)  # damping rate

@inline function sponge_mask(ξ; pad, N)
    ξ ≤ pad         && return (pad + 1 - ξ) / pad
    ξ ≥ N - pad + 1 && return (ξ - (N - pad)) / pad
    return 0.0
end

function du_sponge(i, j, k, grid, clock, fields)
    σx = sponge_mask(i; pad = 8, N = grid.Nx)
    σy = sponge_mask(j; pad = 8, N = grid.Ny)
    -σ₀ * (σx + σy - σx*σy) * fields.U[i, j, k]
end

function dv_sponge(i, j, k, grid, clock, fields)
    σx = sponge_mask(i; pad = 8, N = grid.Nx)
    σy = sponge_mask(j; pad = 8, N = grid.Ny)
    -σ₀ * (σx + σy - σx*σy) * fields.V[i, j, k]
end

forcing = (U = Forcing(du_sponge),
           V = Forcing(dv_sponge))

# ------------------------------
# Model + initial conditions
# ------------------------------
model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = βplane,
                                    buoyancy = buoyancy,
                                    tracers = (:T, :S),
                                    closure = closure,
                                    free_surface = free_surface,
                                    forcing = forcing,
                                    boundary_conditions = bcs)

set!(model, u = 0.0, v = 0.0, T = 18.0, S = 35.5)

# ------------------------------
# Simulation + output
# ------------------------------
Δt = 20.0
stop_time = 2days

isdir("out") || mkpath("out")

ckpt = Checkpointer(model;
    schedule = TimeInterval(6hours),
    dir = "out",
    prefix = "gulf_checkpoint_bathy",
    overwrite_existing = true,
    verbose = true,
)

writers = (ckpt,)

# Optional field writer if JLD2 is available
try
    OW = Oceananigans.OutputWriters
    fw = OW.JLD2OutputWriter(model,
        (; T = model.tracers.T, S = model.tracers.S, η = model.free_surface.η);
        schedule = TimeInterval(12hours),
        filename = "out/gulf_fields_bathy.jld2",
        overwrite_existing = true,
    )
    writers = (ckpt, fw)
catch
    @warn "JLD2 not available; skipping field writer (checkpoints still saved)."
end

simulation = Simulation(model; Δt, stop_time, output_writers = writers)

@info "Starting analytic-bathymetry run…" Δt stop_time
run!(simulation)
@info "Done."
