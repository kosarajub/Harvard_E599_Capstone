# Harvard_E599_Capstone
Capstone project of Data Science ALM at Harvard Extension

## Overview
Above-ground biomass (AGB) is a key indicator of the soil carbon storage potential of blue carbon ecosystems, including salt marshes, seagrasses, and mangroves. These ecosystems play a critical role in global carbon cycling and climate change mitigation, yet accurate large-scale AGB estimation remains challenging due to variability in data quality, sampling methods, and ecosystem types.
This project generates scripts to collect and clean publicly available AGB data and predicts AGB values using Earth Observation (EO) data and machine learning models.

**Key objectives:**
- Generate scripts to collect and clean AGB and EO data
- Extract and process predictor variables from EO datasets (Sentinel-2, EMIT, NISAR)
- Conduct exploratory analysis of AGB relationships with remote sensing predictors
- Develop and evaluate machine learning models for AGB prediction across salt marsh, seagrass, and mangrove ecosystems

**Sponsors:** NASA and the Smithsonian Environmental Research Center (SERC)

---

## Study Area
The current modeling experiments focus primarily on mangrove ecosystems in **Belize**, with the long-term goal of extending predictions to salt marsh and seagrass ecosystems globally.

---

## Data Sources

| Satellite/Dataset | Type | Timeline/Epoch | # Bands/Variables | Source |
|---|---|---|---|---|
| Sentinel-2 | Optical | 2015 → Present (reliable from 2017) | ~13 bands | Google Earth Engine |
| Landsat 8/9 | Optical | 2013 → Present (L8); 2021 → Present (L9) | ~11 bands | Google Earth Engine |
| NISAR | SAR (L + S band) | ~2024/25 → Present | Multiple | NASA Earthdata |
| EMIT | Optical (Hyperspectral) | ~2022 → Present | ~285 spectral bands | NASA Earthdata |
| Simard et al. (2011) | Spaceborne LiDAR + ancillary modeling | Single epoch; Acquired 2005; Published 2011 | 1 variable: canopy height (m); 1 km resolution | Google Earth Engine |
| Simard et al. (2025) | InSAR-derived canopy height, calibrated with GEDI lidar | Single epoch; 2011–2013 | 1 variable: mangrove canopy height (m) | Google Earth Engine |

## Repository Structure

```
├── Data/                          # Root directory for all satellite/sensor AGB datasets used in blue carbon ecosystem modeling
│   ├── EMIT_AGB/                  # AGB predictor variables extracted from NASA's EMIT hyperspectral instrument; PCA-reduced representations used for stable modeling
│   ├── NISAR_AGB/                 # AGB predictor variables from NISAR; note coverage limited to 2025 onwards
│   └── SENTINEL_AGB/              # AGB predictor variables from Sentinel-2; primary optical sensor used across salt marsh, seagrass, and mangrove ecosystems
├── Scripts/                       # All processing, analysis, and modeling code for AGB estimation across coastal blue carbon ecosystems
│   ├── Data_Scripts/              # Scripts for querying, ingesting, and preprocessing raw EO data via cloud-based geospatial APIs
│       ├── EMIT/                  # Extraction and preprocessing scripts for EMIT hyperspectral imagery, including dimensionality reduction via PCA
│       ├── NISAR/                 # Extraction and preprocessing scripts for NISAR SAR backscatter data
│   └── Modeling_Scripts/          # Scripts for spatial modeling, machine learning, and statistical analysis of AGB
│       ├── Analysis/              # Exploratory analysis of AGB-predictor relationships, correlation assessment
│       ├── Common/                # Shared utilities: data loaders, log-transformation helpers, plot-level aggregation
│       ├── Modeling/              # Core modeling code organized by method, evaluating AGB prediction across ecosystem types
│           └── Diffusion/         # Conditional diffusion models for AGB; residual diffusion 
│           └── EMIT/              # Hyperspectral biomass models using PCA-compressed EMIT features
│           └── Kriging/           # Ordinary and Regression Kriging models
│           └── NISAR/             # SAR-based AGB models leveraging NISAR backscatter signals
│           └── Neural Network/    # Deep learning models including multi-modal CNNs
│           └── Results/           # Model outputs, R² scores, residual diagnostics, uncertainty estimates, and performance comparisons across methods
│           └── Sentinel/          # Optical AGB models using Sentinel-2 multispectral features; baseline and residual diffusion experiments conducted here
└── README.md                      # Project overview, setup instructions, and methodology summary for NASA and SERC sponsors
```

## Limitations

- **Sampling bias:** Field biomass data are unevenly distributed across regions and ecosystem types, which may affect model generalizability
- **Synthetic AGB values:** Most published AGB values are derived from allometric equations rather than direct measurement, so model accuracy is partially dependent on the accuracy of those equations
- **NISAR temporal coverage:** NISAR imagery is only available from 2025 onward, limiting alignment with historical field data
- **Geographic scope:** Current experiments are primarily limited to Belize; transferability to other regions has not yet been validated
- **Ecosystem scope:** This study focuses on vegetated coastal ecosystems (salt marshes, seagrasses, mangroves) and does not model below-ground biomass or soil carbon stocks directly

---

## Future Work

- Combine EMIT hyperspectral data with Sentinel-2 and NISAR for multi-modal prediction
- Incorporate ancillary environmental data (DEMs, tidal stage, climate variables) to capture coastal patterns not reflected in EO data alone
- Fine-tune pretrained EO foundation models (e.g., Prithvi-EO-2.0, SatMAE) on plot data instead of training from scratch
- Test geographic generalizability by training on one region and evaluating on another
- Externally validate predictions against independent published AGB products
- Extend modeling to salt marsh and seagrass ecosystems with ecosystem-specific feature analysis

---

## License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
