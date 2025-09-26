using Oceananigans
using Oceananigans.Architectures: CPU, GPU, on_architecture
using JLD2, Printf, Statistics

gridfile = joinpath(@__DIR__, "..", "data", "processed", "ibgrid_gom.jld2")
isfile(gridfile) || error("Missing $(gridfile). Build it first (see preprocess script).")

@load gridfile ibgrid Nx Ny Nz lonW lonE latS latN Hmax
ibgrid = on_architecture(CPU(), ibgrid)

# Show vertical range
zF = Oceananigans.Grids.znodes(ibgrid, Oceananigans.Grids.Face())
zC = Oceananigans.Grids.znodes(ibgrid, Oceananigans.Grids.Center())
@info "Grid size" size=(Nx,Ny,Nz) zF_extrema=(first(zF), last(zF)) zC_extrema=(first(zC), last(zC))

# Make a tiny model and try to write 1.0 to T everywhere wet
model = HydrostaticFreeSurfaceModel(grid=ibgrid, tracers=(:T, :S))
T = model.tracers.T
set!(T, (λ, φ, z) -> 1.0)
Ttop = Array(interior(T))[:,:,end]
wet_frac = count(!iszero, Ttop) / length(Ttop)
@info "Wet fraction (top layer)" wet_fraction=wet_frac

if wet_frac == 0
    @error "No wet cells detected. Rebuild ibgrid (see preprocess script)."
else
    @info "IB grid looks fine."
end
