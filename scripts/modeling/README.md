# Modeling experiments

Light, fast experiments on the warehouse table (`datasets/processed/features/tabular_features.csv`): can the 14 physical
drivers (climate, soil, terrain) reproduce the GAEZ suitability of each crop, and how honest is the evaluation? The
target is GAEZ v5 at 30 arc-seconds (~1 km); `TARGET_RES` in [exp_lib.py](exp_lib.py) switches every notebook back to
the 5 arc-minute (~9 km) product of the first round.

Run the notebooks in order, each one on its own. Every notebook takes a few minutes. The CPU budget is capped at
`N_JOBS = 6` workers in [exp_lib.py](exp_lib.py) (one thread each); change it there if you want more or less.
Small result tables go to `results/`, figures to `figures/`, bulky per-cell predictions to
`datasets/processed/experiments/` (git-ignored). m09 and m10 use the GPU if there is one and fall back to the CPU otherwise.

| notebook | question |
| --- | --- |
| m01 | How well do ridge, kNN, ExtraTrees, HistGradientBoosting and XGBoost recover GAEZ, and how much does random K-fold overstate it? |
| m02 | How does the apparent skill change with the size of the held-out blocks? |
| m03 | Which input groups and variables matter (ablation, SHAP, response curves)? |
| m04 | Do neighbourhood or pixel-mean features help, and what if the model is trained on ~9 km pixel averages? |
| m05 | How well can the recommended-crop label itself be classified? |
| m06 | What happens when a model must extrapolate (other elevation, rainfall, end of the province)? |
| m07 | Which agro-ecological zones emerge without labels, and do they match the crop map? |
| m08 | How do GAEZ and Global Crop Suitability 2024 relate, and do they agree on local detail? |
| m09 | A multi-output MLP on the GPU: a small set of sweeps. |
| m10 | A U-Net on the grid: does spatial context help a deep model? |
| m11 | Does the soil gap-fill matter? |
| m12 | Learning curves and ensembles. |
| m13 | GAEZ ~9 km vs 1 km: what does the finer target change, and was the sub-pixel detail of the ~9 km models real? |
| m14 | Does Sentinel-2 imagery add to the physical drivers (per-cell statistics with trees, image patches with a CNN and a fusion network)? |
| m15 | How sure are the estimates? Adaptive conformal 90 % intervals, P(suitable) and recommendation probabilities per cell. |

`m09` and `m10` read `results/m01_baselines_raw.csv` and the out-of-fold file written by `m01`, so run `m01` first.
Every notebook needs the 1 km columns (`gaez1k_index_<crop>`) written by `scripts/preprocessing/12_gaez_v5_1km.ipynb`:
m01-m12 and m14 learn them on the 9,129 cells with a value for all ten crops, and `m13` compares them with the ~9 km columns
(`gaez_index_<crop>`). `m14` also needs the Sentinel-2 cell statistics and patches from
`scripts/preprocessing/11_sentinel2_cell_features.ipynb`.
Models: `exp_lib.py` (scikit-learn / XGBoost) and `exp_torch.py` (PyTorch). Always `import torch` before `shap`
(see the docstring of `exp_lib.py`).
