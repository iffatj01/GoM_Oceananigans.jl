# GPU-Accelerated Free-Running Gulf of Mexico Simulation

This repository contains Julia code and configurations for a **high-resolution (1/25°) free-running simulation of the Gulf of Mexico** using [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl).  
The focus is on **Loop Current dynamics, eddy shedding**, and **benchmarking GPU performance** relative to established models such as HYCOM, NEMO, and MITgcm.

---

##  Project Highlights
- **GPU-accelerated**: ~15 simulated years/day on 4× NVIDIA A100 GPUs.  
- **Resolution**: 1/25° horizontal grid with realistic bathymetry.  
- **Target circulation features**:
  - Loop Current variability  
  - Eddy shedding events  
  - Yucatán Channel transport  
- **Validation datasets**:
  - NOAA OISST (sea surface temperature)  
  - Salinity climatologies  
  - Satellite altimetry (eddy positions)  
  - NeSPReSO synthetic profiles (subsurface stratification)  

---

## Repository Structure

```text
preprocess/                     # Preprocessing scripts (bathymetry, masks, forcing)
runs/                           # Main model configurations & run scripts
scripts/                        # Utility scripts (validation, plotting, data handling)
gulf_simulation.jl              # Example GoM simulation setup
quicklook_05_seasonal.jl        # Quicklook seasonal forcing test case
test_code.jl                    # Testbed / sandbox code
Project.toml                    # Julia environment definition
.gitignore                      # Git ignore rules
ocean_wind_mixing_and_convection.mp4   # Example visualization
two_dimensional_turbulence.mp4         # Example visualization
README.md                       # Project documentation

---
```
##  Validation Framework
Model skill is assessed using **community-standard benchmarks**:
- **SST bias**: Typically < 1 °C (slightly larger on shallow shelves).  
- **Salinity error**: < 1 PSU over most of the domain.  
- **Loop Current & eddy shedding**: Compared against satellite-derived statistics.  
- **Subsurface thermal/haline structure**: Evaluated with NeSPReSO profiles.  
- **Transport metrics**: Yucatán Channel transport within observational constraints.  
- **Skill metrics**: RMSE, mean bias, and Modified Hausdorff Distance for frontal diagnostics.  

---

##  Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/iffatj01/GoM_Oceananigans.jl.git
   cd GoM_Oceananigans.jl```
2. Start Julia and activate the environment:
 ```bash
julia --project=.
```
3. Install dependencies
```bash
using Pkg
Pkg.instantiate()
```
## Running a Simulation
Example: Run the Gulf of Mexico seasonal spin-up
``` bash
julia runs/06_spinup_ibgrid_seasonal.jl
```
Quicklook test:
```bash
julia quicklook_05_seasonal.jl
```
## Future Work

- Modal decomposition (e.g., Hilbert EOFs) to analyze vertical coupling.  
- Improved atmosphere–ocean fluxes (possibly using [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl)).  
- Deeper validation of baroclinic processes and internal tides.  


