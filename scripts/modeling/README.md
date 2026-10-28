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
| m16 | The same models with and without a Big Data framework: Spark MLlib pipelines vs scikit-learn (ridge, random forest, gradient boosting) on identical folds, plus training time against data size. |
| e01 | Evaluation only: every model on one set of metrics, per-crop metrics from the out-of-fold predictions, error by land cover, Wilcoxon tests. |
| e02 | Evaluation only: m16 with vs without Spark MLlib, paired per crop, and a power-law fit of the scaling test. |
| e03 | Evaluation only: g01 graph results re-checked from the saved node table, with the insight summary. |

The e notebooks only read `results/` and `datasets/processed/experiments/`, so they run in seconds once m01-m16 and g01
have been run.
| g01 | Graph analytics on the 1 km grid graph: contiguous suitable land and its bottleneck cells, recommendation blocks, Louvain communities, the crop competition and nesting graph, and the same graph on Spark with GraphFrames. |

`m09` and `m10` read `results/m01_baselines_raw.csv` and the out-of-fold file written by `m01`, so run `m01` first.
Every notebook needs the 1 km columns (`gaez1k_index_<crop>`) written by `scripts/preprocessing/12_gaez_v5_1km.ipynb`:
m01-m12 and m14 learn them on the 9,129 cells with a value for all ten crops, and `m13` compares them with the ~9 km columns
(`gaez_index_<crop>`). `m14` also needs the Sentinel-2 cell statistics and patches from
`scripts/preprocessing/11_sentinel2_cell_features.ipynb`.
`m16` and `g01` start a local Spark session through `exp_spark.py` (local mode, 6 cores, the Windows settings of the
Sedona notebook); `g01` fetches the GraphFrames package (`graphframes:graphframes:0.8.4-spark3.5-s_2.12`) from the
Spark Packages repository the first time, so it needs internet once. `g01` writes the per-cell graph attributes to
`datasets/processed/experiments/g01_nodes.parquet`, read by the dashboard's Graph Analytics page.
Models: `exp_lib.py` (scikit-learn / XGBoost), `exp_torch.py` (PyTorch) and `exp_spark.py` (Spark MLlib, GraphFrames).
Always `import torch` before `shap` (see the docstring of `exp_lib.py`).
