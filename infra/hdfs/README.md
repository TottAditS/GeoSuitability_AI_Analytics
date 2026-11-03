# HDFS

HDFS for the Spark layer: one NameNode and two DataNodes, replication 2, no YARN. Every block is stored on both DataNodes, so losing one loses no data. It uses about 2.3 GB of RAM while running. The local `datasets/` folder stays the source of truth.

## Setup

1. Install Docker Desktop with the WSL 2 backend.
2. Optional: cap WSL memory in `%UserProfile%\.wslconfig`:

   ```
   [wsl2]
   memory=4GB
   processors=2
   ```

3. Start HDFS and copy the data in (it also starts Docker Desktop if needed):

   ```powershell
   powershell -ExecutionPolicy Bypass -File infra/hdfs/load_lake.ps1
   ```

4. Check that Spark can read and write it:

   ```powershell
   python infra/hdfs/verify_hdfs.py
   ```

The NameNode UI is at http://localhost:9870.

## Start and stop

HDFS only runs when you start it. Nothing starts at login.

The easy way is the notebook [hdfs_launcher.ipynb](hdfs_launcher.ipynb), like the dashboard launcher: run the first cell to start, run the last cell to stop. It also has optional checks (a Spark test and a backup test).

From a terminal:

```powershell
powershell -ExecutionPolicy Bypass -File infra/hdfs/start_hdfs.ps1   # opens Docker Desktop if needed, about 45 s from cold
powershell -ExecutionPolicy Bypass -File infra/hdfs/stop_hdfs.ps1    # stops HDFS and closes Docker Desktop, a few seconds
```

`stop_hdfs.ps1 -KeepDocker` leaves Docker Desktop open. The data survives stop and start. `docker compose down` deletes it, so run `load_lake.ps1` again after that.

## How code picks HDFS or local

`lake_path()` (in `scripts/download/common.py` and `scripts/modeling/exp_spark.py`) returns an HDFS URI while the NameNode answers on `localhost:9000`, and a path under `datasets/` otherwise. The same code runs either way.

`LAKE_URI` overrides this. Set it to a URI to use another HDFS, or to `local` to force the local disk.

## What uses it

- Notebook `06_sedona_processing` reads the fact table from HDFS and writes `spark_output` back.
- The dashboard (Data Lake & Warehouse, Spark / Sedona tab) reads `spark_output` through WebHDFS. If HDFS is down, it reads the local copy.
- `m16` and `g01` do not use it. They build their Spark input from pandas so the folds match scikit-learn.

## What is copied

| HDFS path | From |
|---|---|
| `/lake/processed/features/*.csv` | `datasets/processed/features` |
| `/lake/processed/experiments/` | `datasets/processed/experiments` |
| `/lake/processed/spark_output/` | `datasets/processed/spark_output`, without `checkpoints/` |
| `/lake/processed/grid/grid_1km.gpkg` | `datasets/processed/grid` |
| `/lake/raw_sample/data_dictionary.csv` | `datasets/data_dictionary.csv` |

`/lake/tmp/` holds the scratch output of `verify_hdfs.py`.

Not copied: `datasets/raw` (26 GB of rasters that GDAL reads from disk), the Sentinel-2 patches and the mosaics.

## Notes

- The image runs Hadoop as the `hadoop` user, so data lives in `/tmp/hadoop-hadoop` inside the containers, not in a Docker volume. That is why `down` wipes it.
- Both DataNodes advertise themselves as `localhost`. The first publishes ports 9864 and 9866, the second 9875 and 9876. Clients on Windows set `dfs.client.use.datanode.hostname=true` (`exp_spark.py` does) and need no hosts-file edit. Clients inside the containers use the container IP.
- The `apache/hadoop:3.3.6` image matches the Hadoop 3.3 client bundled with PySpark 3.5.
