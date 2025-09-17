# runs/06_spinup_ibgrid_seasonal.jl
using Oceananigans
using Oceananigans.Units: second, seconds, minute, minutes, hour, hours, day, days
using JLD2
using Printf

# ----------------- pick architecture -----------------
function pick_architecture(dev::Int=0)
    try
        @eval using CUDA
        if CUDA.has_cuda()
            CUDA.device!(dev)
            return GPU()
        end
    catch
        # no CUDA; fall through
    end
    return CPU()
end

arch = pick_architecture()
@info "Running on $(arch isa GPU ? "GPU" : "CPU") architecture"

# ----------------- load IB grid -----------------
proc_jl = joinpath(@__DIR__, "..", "data", "processed", "ibgrid_gom.jld2")
isfile(proc_jl) || error("Missing $(proc_jl). Run preprocess/make_ibgrid_from_bathy.jl first.")
@load proc_jl ibgrid Hmax lonW lonE latS latN Nx Ny Nz

# normalize to CPU (handles files saved on GPU) then move to chosen arch
ibgrid = Oceananigans.Architectures.on_architecture(CPU(), ibgrid)
ibgrid = Oceananigans.Architectures.on_architecture(arch, ibgrid)

# ----------------- physics -----------------
ϕ₀ = 0.5 * (latS + latN)
Ω  = 7.2921159e-5
f0 = 2Ω * sind(ϕ₀)

buoy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(
    thermal_expansion  = 2.0e-4,
    haline_contraction = 7.5e-4,
))
cori    = FPlane(f = f0)
closure = AnisotropicMinimumDissipation()

model = HydrostaticFreeSurfaceModel(;
    grid         = ibgrid,
    tracers      = (:T, :S),
    buoyancy     = buoy,
    coriolis     = cori,
    closure      = closure,
    free_surface = SplitExplicitFreeSurface(),
)

# ----------------- initial conditions -----------------
T_init(x, y, z) = 20.0 + 4.0 * exp(z / 100.0)  # z ≤ 0
S_init(x, y, z) = 35.0
set!(model; u = 0.0, v = 0.0, T = T_init, S = S_init)

# ----------------- output setup -----------------
const OW = Oceananigans.OutputWriters
outdir = joinpath(@__DIR__, "..", "outputs"); mkpath(outdir)

u, v, _w = model.velocities
T, S     = model.tracers
η        = model.free_surface.η

# progress print
function progress(sim)
    t  = sim.model.clock.time
    it = iteration(sim)
    @printf(" iter=%6d  t=%10.1f s (%.2f d)  Δt=%.1f s\n", it, t, t/86400, sim.Δt)
end

# Try built-in JLD2 writer, else manual snapshots via callback
has_jld2writer = isdefined(OW, :JLD2OutputWriter)
if has_jld2writer
    jld2_writer = OW.JLD2OutputWriter(model, (; u, v, T, S, η);
        schedule           = TimeInterval(6hours),
        filename           = joinpath(outdir, "spinup_ibgrid_seasonal.jld2"),
        overwrite_existing = true,
    )
else
    @warn "JLD2OutputWriter not available in this Oceananigans build; using manual JLD2 snapshots."
    function manual_save(sim)
        it = iteration(sim)
        t  = sim.model.clock.time
        fn = joinpath(outdir, @sprintf("snap_%07d.jld2", it))
        u_cpu  = Array(interior(u))
        v_cpu  = Array(interior(v))
        T_cpu  = Array(interior(T))
        S_cpu  = Array(interior(S))
        η_cpu  = Array(interior(η))
        @save fn t u_cpu v_cpu T_cpu S_cpu η_cpu
    end
end

# Checkpointer if available
has_checkpointer = isdefined(OW, :Checkpointer)
if has_checkpointer
    ckpt = OW.Checkpointer(model;
        schedule = TimeInterval(1day),
        prefix   = joinpath(outdir, "chkpt_spinup"),
        overwrite_existing = true,
    )
end

# ----------------- simulation -----------------
Δt        = 300seconds
stop_time = 5days
simulation = Simulation(model; Δt, stop_time)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

if has_checkpointer
    simulation.output_writers[:checkpoint] = ckpt
end

if has_jld2writer
    simulation.output_writers[:jld2] = jld2_writer
else
    simulation.callbacks[:manual_save] = Callback(manual_save, TimeInterval(6hours))
end

@info "Starting run…  Nx×Ny×Nz=$(Nx)×$(Ny)×$(Nz)  Δt=$(Δt)  stop=$(stop_time)"
run!(simulation)
@info "Done. Outputs at: $(outdir)"

