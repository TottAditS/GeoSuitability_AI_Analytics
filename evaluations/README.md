# Evaluations

One folder with every evaluation notebook of the project, grouped by what it evaluates: the data (EDA, dataset quality),
the models (machine learning, Spark MLlib) and the graph analytics. Meant as a single place to open for grading or review.

| notebook | evaluates | source of truth |
| --- | --- | --- |
| [01_eda_evaluation.ipynb](01_eda_evaluation.ipynb) | the warehouse table: distributions, correlations, spatial patterns, GAEZ class balance | `scripts/preprocessing/14_eda_current_warehouse.ipynb` |
| [02_dataset_quality_evaluation.ipynb](02_dataset_quality_evaluation.ipynb) | every raw dataset: CRS, resolution, extent, nodata, value ranges | `scripts/preprocessing/01_validate_datasets.ipynb` |
| [03_ml_model_evaluation.ipynb](03_ml_model_evaluation.ipynb) | machine learning model performance: leaderboard, per-crop metrics, errors, significance tests, uncertainty | `scripts/modeling/e01_ml_model_evaluation.ipynb` |
| [04_spark_mllib_evaluation.ipynb](04_spark_mllib_evaluation.ipynb) | models with vs without Spark MLlib: accuracy, training time, scalability | `scripts/modeling/e02_spark_mllib_evaluation.ipynb` |
| [05_graph_analytics_evaluation.ipynb](05_graph_analytics_evaluation.ipynb) | graph analytics results: structure, patches, communities, crop competition, NetworkX vs GraphFrames | `scripts/modeling/e03_graph_analytics_results.ipynb` |

**These are duplicate copies, not the maintained files.** Each one is generated from the "source of truth" notebook listed
above, with only its path-resolution cell adjusted so it still runs correctly when opened from this folder instead of its
original directory. If the pipeline changes, edit the source notebook and regenerate its copy here — don't edit a copy in
this folder directly, or the two will drift apart.

All five run top to bottom from this folder, in any order, independently of each other. 01 and 02 read the data lake /
warehouse directly (`datasets/raw/`, `datasets/processed/`) and take a few minutes; 03-05 only read the small result
tables already written to `scripts/modeling/results/` and run in seconds.
