#!/usr/bin/env julia
using NCDatasets
using Printf
using Dates

# --------- area you want (same box as your runs) ----------
const lonW, lonE = -98.0, -80.0
const latS, latN =  18.0,  31.0

# --------- output path ------------------------------------
const out_nc = joinpath(@__DIR__, "..", "data", "raw", "bathymetry.nc")
mkpath(dirname(out_nc))

# --------- remote ETOPO1 (bedrock) OPeNDAP ----------------
const url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO1_Bed_g_gmt4.nc"

@info "Opening remote ETOPO1…" url
ds = NCDataset(url)

# Axis names vary; be forgiving:
lon = haskey(ds, "x")         ? vec(ds["x"][:]) :
      haskey(ds, "lon")       ? vec(ds["lon"][:]) :
      haskey(ds, "longitude") ? vec(ds["longitude"][:]) :
      error("Could not find a longitude variable.")

lat = haskey(ds, "y")          ? vec(ds["y"][:]) :
      haskey(ds, "lat")        ? vec(ds["lat"][:]) :
      haskey(ds, "latitude")   ? vec(ds["latitude"][:]) :
      error("Could not find a latitude variable.")

zname = haskey(ds, "z") ? "z" : haskey(ds, "elevation") ? "elevation" :
         haskey(ds, "Band1") ? "Band1" :
         error("Could not find the elevation/topography variable.")

# Helpers to make slicing robust to ascending/descending axes
axis_desc(a) = length(a) > 1 && a[2] < a[1]
function index_range(arr, a, b)
    desc = axis_desc(arr)
    work = desc ? reverse(arr) : arr
    i1 = searchsortedfirst(work, a)
    i2 = searchsortedlast(work,  b)
    (i1 <= i2) || error("Range produced empty subset.")
    return desc ? ((length(arr)-i2+1):(length(arr)-i1+1)) : (i1:i2)
end

# Does the dataset use 0..360 or -180..180?
is0360 = minimum(lon) >= 0            # true for 0..360; false for -180..180
wrap360(x) = x < 0 ? x + 360 : x

# Build longitude indices
if is0360
    a = wrap360(lonW)
    b = wrap360(lonE)
    if a <= b
        I = index_range(lon, a, b)
    else
        # If your box crosses the dateline; not the case here but keep it robust
        I1 = index_range(lon, a, maximum(lon))
        I2 = index_range(lon, minimum(lon), b)
        I  = vcat(collect(I1), collect(I2))  # contiguous union
    end
else
    I = index_range(lon, lonW, lonE)
end

# Latitude indices
J = index_range(lat, latS, latN)

# Read subset (ETOPO1 is elevation +up). Output should be depth +down.
Zsub = ds[zname][J, I]
close(ds)

# Put axes in ascending order for output NetCDF
lon_out = lon[I]
lat_out = lat[J]
if axis_desc(lat)
    Zsub    = reverse(Zsub, dims=1)
    lat_out = reverse(lat_out)
end
if axis_desc(lon)
    Zsub    = reverse(Zsub, dims=2)
    lon_out = reverse(lon_out)
end

depth = -Array(Zsub)  # convert +up elevation to +down depth

@info @sprintf("Cutout dims: lat=%d, lon=%d", size(depth, 1), size(depth, 2))
@info @sprintf("lat: %.3f..%.3f | lon: %.3f..%.3f",
               first(lat_out), last(lat_out), first(lon_out), last(lon_out))

# ---- write compact local NetCDF
dsout = NCDataset(out_nc, "c",
    attrib = Dict(
        "title"        => "ETOPO1 subset for GoM",
        "source"       => url,
        "history"      => "Created $(Dates.now())",
        "Conventions"  => "CF-1.8",
    )
)
defDim(dsout, "lon", length(lon_out))
defDim(dsout, "lat", length(lat_out))
v_lon = defVar(dsout, "lon", Float32, ("lon",), attrib = Dict("units"=>"degrees_east"))
v_lat = defVar(dsout, "lat", Float32, ("lat",), attrib = Dict("units"=>"degrees_north"))
v_z   = defVar(dsout, "z",   Float32, ("lat","lon"),
               attrib = Dict("long_name"=>"ocean depth", "units"=>"m", "positive"=>"down"))
v_lon[:]  = Float32.(lon_out)
v_lat[:]  = Float32.(lat_out)
v_z[:, :] = Float32.(depth)
close(dsout)

@info "Wrote → $out_nc"
