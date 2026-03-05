# LPBF-DEM-CFD-ML

**Data-Driven Analysis of Molten Pool Morphology and Porosity Formation in Laser Powder Bed Fusion via Integrated DEM-CFD and Machine Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> **Paper:** "Data-Driven Analysis of Molten Pool Morphology and Porosity Formation in Laser Powder Bed Fusion via Integrated DEM-CFD and Machine Learning"
> **Journal:** *Powder Technology* (under review)
> **Authors:** [Xin He, Xiaoming Wang, Corey Vian]
> **DOI:** [TBD]

---

## Repository Contents

```
LPBF-DEM-CFD-ML/
│
├── README.md                                  ← This file
├── requirements.txt                           ← Python dependencies
├── LICENSE                                    ← MIT License
│
├── data/
│   └── Melt_Pool_Data_With_Depth.xlsx         ← Full ML dataset (234 samples)
│
├── dem_lpbf/                                  ← DEM powder-bed simulation (Fig. 2)
│   ├── LPBF_PowderGeneration.liggghts_init    ← LIGGGHTS 3.8.0 input script
│   ├── bed19.stl                              ← Powder-bed substrate geometry
│   ├── recoater16.stl                         ← Recoater blade geometry
│   └── Insertionsface11.stl                   ← Particle insertion plane
│
├── fluent_udf/
│   └── lpbf_gaussian_heat_source.c            ← ANSYS Fluent UDF (CFD, Figs. 3, 7, 9–12)
│
├── ml_models/                                 ← ML model scripts (Figs. 13, 14)
│   ├── model_GPR.py                           ← Gaussian Process Regression (Fig. 13a)
│   ├── model_SVR.py                           ← Support Vector Regression (Fig. 13b)
│   ├── model_BPNN.py                          ← Backpropagation Neural Network (Fig. 13c)
│   ├── model_RandomForest.py                  ← Random Forest (Fig. 13d)
│   ├── model_XGBoost.py                       ← XGBoost (Fig. 13e)
│   ├── model_BayesianRidge.py                 ← Bayesian Ridge Regression (Fig. 13f)
│   ├── model_LightGBM.py                      ← LightGBM (Fig. 13g)
│   ├── model_CatBoost.py                      ← CatBoost (Fig. 13h)
│   └── compare_models_Fig14.py                ← Model comparison (Fig. 14)
│
├── NSGA-II optimization/                      ← NSGA-II optimization (Fig. 15)
│   ├── NSGA2.py                               ← NSGA-II optimization (Fig. 15)
└── figure_scripts/                            ← Additional figure scripts
    ├── fig05_CDF_consistency_check.py         ← Fig. 5: CDF validation of data
    └── fig16_correlation_heatmap.py           ← Fig. 16: Pearson correlation heatmap
```

---

## Dataset

**File:** `data/Melt_Pool_Data_With_Depth.xlsx`
**Samples:** 234 (34 baseline CFD configurations × data augmentation via GPR)
**Source:** ANSYS Fluent 2022 R2 CFD simulations of 18Ni300 maraging steel LPBF

| Column | Unit | Description |
|---|---|---|
| Power [W] | W | Laser power |
| Velocity [mm s-1] | mm/s | Laser scan speed |
| Radius [mm] | mm | Laser beam radius |
| Max Melt Pool Width [mm] | mm | Maximum molten pool width |
| Max Melt Pool Depth [mm] | mm | Maximum molten pool depth |
| Porosity [%] | % | Predicted porosity fraction |

---

## Quick Start

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/jikemuku1/LPBF-DEM-CFD-ML.git
cd LPBF-DEM-CFD-ML
pip install -r requirements.txt
```

---

## DEM Simulation — Powder Bed Generation (Fig. 2)

**Software:** LIGGGHTS 3.8.0 (open-source DEM, https://www.cfdem.com)

All geometry STL files are included in `dem_lpbf/`.

```bash
cd dem_lpbf
liggghts < LPBF_PowderGeneration.liggghts_init
```

This creates a `post/` directory containing:
- `post/particles_*.vtk` — particle positions and kinematics at each dump step
- `post/Bed*.stl` / `post/Recoater*.stl` — geometry snapshots

**Visualisation:** Open `post/particles_*.vtk` in ParaView to reproduce Fig. 2.

**Import to Fluent (Fig. 3):** Export the final particles from ParaView and import them into ANSYS Fluent.

---

## CFD Simulation — Melt Pool (Figs. 3, 7, 9–12)

**Software:** ANSYS Fluent 2022 R2 — VOF multiphase, pressure-based solver

**Load the UDF in Fluent:**
```
Define > User-Defined > Functions > Compiled
Source file: fluent_udf/lpbf_gaussian_heat_source.c
Build → Load
```

**Assign hooks in Fluent:**
| Hook type | UDF name |
|---|---|
| ADJUST | `adjust_gradient` |
| Energy source (metal phase) | `heat_source` |
| X-momentum source | `x_pressure` |
| Y-momentum source | `y_pressure` |
| Z-momentum source | `z_pressure` |

4 User-Defined Memory (UDM) slots must be allocated before loading:
`Define > User-Defined > Memory → 4`

**Physical models implemented:**
- Conical Gaussian laser irradiance with beam-radius taper
- Convective and radiative surface heat losses
- Recoil pressure (Hertz-Knudsen model, Anisimov back-flux correction)
- Marangoni convection (set via Fluent surface-tension gradient panel)
- Temperature-dependent thermophysical properties for 18Ni300

---

## ML Experiments — Reproduce Figures 13 & 14

All scripts use `../data/Melt_Pool_Data_With_Depth.xlsx` (relative path).
Each script saves its output figure as a `.png` file in the working directory.

```bash
cd ml_models

python model_GPR.py           # Fig. 13a → Fig13a_GPR_CrossVal_Results.png
python model_SVR.py           # Fig. 13b → Fig13b_SVR_CrossVal_Results.png
python model_BPNN.py          # Fig. 13c → Fig13c_BPNN_CrossVal_Results.png
python model_RandomForest.py  # Fig. 13d → Fig13d_RandomForest_CrossVal_Results.png
python model_XGBoost.py       # Fig. 13e → Fig13e_XGBoost_CrossVal_Results.png
python model_BayesianRidge.py # Fig. 13f → Fig13f_BayesianRidge_CrossVal_Results.png
python model_LightGBM.py      # Fig. 13g → Fig13g_LightGBM_CrossVal_Results.png
python model_CatBoost.py      # Fig. 13h → Fig13h_CatBoost_CrossVal_Results.png

python compare_models_Fig14.py          # Fig. 14 → Fig14_Model_Comparison.png
python NSGA2.py               # Fig. 15 → Fig_NSGA2_Pareto.png
```

### Reproduce additional figures

```bash
cd figure_scripts
python fig05_CDF_consistency_check.py   # Fig. 5  → Fig5_Consistency_Comparison.png
python fig16_correlation_heatmap.py     # Fig. 16 → Fig16_Correlation_Heatmap.png
```

---

## Reproducibility Notes

| Item | Setting |
|---|---|
| Random seed | `random_state=42`, `np.random.seed(42)` throughout |
| Cross-validation | Honest 10-fold `cross_val_predict()` — predictions only on held-out folds |
| Normalization | `MinMaxScaler` fit on full dataset before CV |
| Figure plots | All three targets (Width, Depth, Porosity) always shown unconditionally |
| Python version | 3.9+ recommended |
| ANSYS Fluent | 2022 R2 |
| LIGGGHTS | 3.8.0 |

---

## Figures Not Reproducible from Code

| Figure | Source |
|---|---|
| Fig. 1 | Optical microscopy — experimental image |
| Fig. 4 | ANSYS Fluent post-processor contour plots |
| Fig. 6 | SEM — experimental image |
| Fig. 8 | Statistical box plots (Origin software) |
| Fig. 15 | NSGA-II Pareto optimisation |
| Fig. 17 | Verification |
| Fig. 18 | 3-D contour maps — script to be added |

---

## Citation

```bibtex
@article{
  title   = {Data-Driven Analysis of Molten Pool Morphology and Porosity Formation
             in Laser Powder Bed Fusion via Integrated DEM-CFD and Machine Learning},
  author  = {[Xin He, Xiaoming Wang, Corey Vian]},
  journal = {Powder Technology},
  year    = {2025},
  doi     = {[DOI: TBD]}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

For questions please open a GitHub Issue or contact:
he730@purdue.edu
