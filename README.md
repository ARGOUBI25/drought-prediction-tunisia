# Spatiotemporal SHAP Analysis for Agricultural Drought Prediction
## A Multi-Source Machine Learning Framework in Semi-Arid Tunisia

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18733246.svg)](https://doi.org/10.5281/zenodo.18733246)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Frontiers in AI](https://img.shields.io/badge/Journal-Frontiers%20in%20AI-red.svg)](https://www.frontiersin.org/journals/artificial-intelligence)

---

## 📌 Overview

This repository contains the full reproducible codebase for the research paper:

> **Spatiotemporal SHAP Analysis for Agricultural Drought Prediction: A Multi-Source Machine Learning Framework in Semi-Arid Tunisia** > *Frontiers in Artificial Intelligence — Section: AI in Food, Agriculture and Water* > 
> **Authors:** Argoubi Majdi, [Co-authors names]  
> **Paper DOI:** [Pending Publication]  
> **Dataset DOI:** [10.5281/zenodo.18733246](https://doi.org/10.5281/zenodo.18733246)

### Study Area
Five semi-arid governorates of central Tunisia: **Kairouan, Kasserine, Sidi Bou Zid, Gafsa, and Siliana**.

### Objectives
1. Predict agricultural drought (SSMI) using multi-source environmental and socioeconomic data.
2. Evaluate 6 ML/DL models (XGBoost, LightGBM, CatBoost, RF, BPNN, LSTM) with a focus on extreme event generalization (2021–2022).
3. Use **Tree SHAP** to explain global drivers and spatiotemporal drought dynamics.

---

## Key Results

| Model | Test R² | Test SDI | Val R² (2021–22) | Val SDI |
|-------|---------|----------|------------------|---------|
| **BPNN** | **0.860** | **0.626** | 0.655 | 0.421 |
| XGBoost | 0.817 | 0.572 | **0.696** | **0.448** |
| LightGBM | 0.807 | 0.561 | — | — |
| LSTM | 0.790 | 0.541 | 0.667 | — |
| CatBoost | 0.756 | — | — | — |
| RF | 0.681 | — | — | — |

**Top SHAP predictors (XGBoost):**
- MERRA-2 Surface SM: 26.0%
- Temperature (T2M): 14.2%
- Sand content: 10.0%
- Precipitation: 8.6%

---

## Repository Structure

```
drought-prediction-tunisia/
├── data/               # Raw and processed data (see data/README.md)
├── notebooks/          # Jupyter notebooks (exploration → figures)
├── src/                # Python source modules
│   ├── data/           # Download and preprocessing scripts
│   ├── models/         # Training, evaluation, metrics
│   ├── shap/           # SHAP global, dependence, spatiotemporal
│   └── figures/        # Figure generation scripts
├── results/            # CSV outputs (metrics, predictions, SHAP values)
└── figures/            # Final figures (PNG 300 dpi)
```

---

## Data Sources

Processed Dataset
The final assembled dataset (154,704 pixel-month observations) is archived on Zenodo:
[10.5281/zenodo.18733246](https://doi.org/10.5281/zenodo.18733246)

| Variable | Source | Resolution | Period |
|----------|--------|------------|--------|
| SM (target SSMI) | GLEAM v4.2a | 0.05° / daily | 1980–2022 |
| SM, T2M, PRECTOTCORR, RH2M, EVPTRNS | NASA POWER MERRA-2 | 0.5° / monthly | 2001–2022 |
| Clay, sand, PAWC, soil depth | SoilGrids v2.0 | 250 m | static |
| Elevation, slope, TWI | SRTM DEM | 90 m | static |
| Land use, population density | ESA WorldCover + WorldPop | 10 m / 100 m | 2020 |

> ⚠️ **Raw data are not versioned** (file size). Download instructions are
> provided in [`data/README.md`](data/README.md).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/drought-prediction-tunisia.git
cd drought-prediction-tunisia

# Create conda environment
conda env create -f environment.yml
conda activate drought-tunisia

# Or with pip
pip install -r requirements.txt
```

---

## Reproducibility

Run the full pipeline in order:

```bash
# 1. Download and preprocess data
python src/data/download_nasa_power.py
python src/data/download_gleam.py
python src/data/compute_ssmi.py
python src/data/merge_dataset.py

# 2. Train and evaluate all six models
python src/models/train_models.py
python src/models/evaluate_models.py

# 3. SHAP analysis
python src/shap/global_importance.py
python src/shap/dependence_plots.py
python src/shap/spatiotemporal_shap.py

# 4. Generate all figures
python src/figures/generate_figure1_studyarea.py
python src/figures/generate_figure2_climate.py
# ... etc.
```

Or run the Jupyter notebooks sequentially:
```
notebooks/01 → 02 → 03 → 04 → 05
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{Argoubi2025drought,
  title   = {Spatiotemporal {SHAP} Analysis for Agricultural Drought 
             Prediction: A Multi-Source Machine Learning Framework 
             in Semi-Arid {Tunisia}},
  author  = {[Author(s)]},
  journal = {Frontiers in Artificial Intelligence},
  year    = {2026},
  doi     = {[to be added]}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
