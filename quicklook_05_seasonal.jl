# quicklook_05_seasonal.jl
# Small sanity-check plots + a tiny movie from out/gulf_seasonal_fields.jld2

using Oceananigans
using JLD2
using CairoMakie

CairoMakie.activate!()  # good for headless servers

# ---- where the JLD2 was written by 05_surface_seasonal.jl
root = @__DIR__
filepath = joinpath(root, "out", "gulf_seasonal_fields.jld2")
@info "Reading: $filepath"
isfile(filepath) || error("Couldn't find $filepath. Did the seasonal run write it?")

# Load time series lazily
tsT = FieldTimeSeries(filepath, "T")
times = tsT.times
N = length(times)
N > 0 || error("No time slices found in $filepath")

# Helper: move data (CPU/GPU) to a plain Array
to_host(A) = Array(interior(A))

# --- 1) Surface T snapshot (last time)
T3_last = to_host(tsT[N])           # (Nx, Ny, Nz)
Nx, Ny, Nz = size(T3_last)
ksurf = Nz                          # top-most scalar cell
T_sfc_last = T3_last[:, :, ksurf]

fig1 = Figure(size=(900, 700))
ax1  = Axis(fig1[1,1], title = "Surface T (last time = $(round(times[end]/86400, digits=2)) d)",
            xlabel = "x index", ylabel = "y index")
heatmap!(ax1, T_sfc_last')          # transpose for x-right, y-up display
save(joinpath(root, "out", "quicklook_T_surface_last.png"), fig1)
@info "Saved: out/quicklook_T_surface_last.png"

# --- 2) Tiny animation of surface T
frames = 1:clamp(N, 1, 100)         # keep it short
fig2 = Figure(size=(900, 700))
ax2  = Axis(fig2[1,1], title = "Surface T — t=$(round(times[1]/86400, digits=2)) d",
            xlabel = "x index", ylabel = "y index")
data = Observable(T_sfc_last')
heatmap!(ax2, data)

mp4path = joinpath(root, "out", "quicklook_T_surface.mp4")
record(fig2, mp4path, frames; framerate=10) do i
    Ti = to_host(tsT[i])
    data[] = Ti[:, :, ksurf]'       # update image
    ax2.title = "Surface T — t=$(round(times[i]/86400, digits=2)) d"
end
@info "Saved: $(mp4path)"
